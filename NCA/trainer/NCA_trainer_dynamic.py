import jax
import jax.numpy as jnp
import jax.random as jr
import optax
import equinox as eqx
import datetime
import Common.trainer.loss as loss
import jaxpruner
from NCA.trainer.NCA_trainer import NCA_Trainer
from functools import partial
from NCA.trainer.tensorboard_log import NCA_Train_log, kaNCA_Train_log, mNCA_Train_log, aNCA_Train_log
from NCA.model.NCA_KAN_model import kaNCA
from NCA.model.NCA_multi_scale import mNCA
from NCA.model.NCA_multihead_attention import aNCA
from NCA.trainer.data_augmenter_nca import DataAugmenter
from einops import repeat
from Common.utils import key_pytree_gen
from Common.model.boundary import model_boundary, hard_boundary, no_boundary
from tqdm import tqdm
from jaxtyping import Float,Array,Key
import time

class NCA_Trainer_dynamic(NCA_Trainer):
	def __init__(self,
                 dt_seq,
                 *args,
                 **kwargs):
		super().__init__(*args,**kwargs)
		self.dt_seq = dt_seq

	def train(self,
		      t,
			  iters,
			  optimiser=None,
			  STATE_REGULARISER=1.0,
			  BOUNDARY_REGULARISER=1.0,
			  WARMUP=64,
			  LOG_EVERY=40,
			  CLEAR_CACHE_EVERY=100,
			  WRITE_IMAGES=True,
			  LOSS_FUNC_STR = "euclidean",
			  LOOP_AUTODIFF = "checkpointed",
			  SPARSE_PRUNING = False,
			  TARGET_SPARSITY = 0.5,
			  wandb_args={"project":"NCA",
				 		  "group":"group_1",
				 		  "tags":["training"]},
			  key=jr.PRNGKey(int(time.time()))):
		"""
		Perform t steps of NCA on x, compare output to y, compute loss and gradients of loss wrt model parameters, and update parameters.

		Parameters
		----------
		t : int
			number of NCA timesteps between x[N] and x[N+1]
		iters : int
			number of training iterations
		optimiser : optax.GradientTransformation
			the optax optimiser to use when applying gradient updates to model parameters.
			if None, constructs adamw with exponential learning rate schedule
		STATE_REGULARISER : float optional
			Strength of intermediate state regulariser. Defaults to 1.0
		WARMUP : int optional
			Number of iterations to wait for until starting model checkpointing
		LOG_EVERY : int optional
			Save output of model every LOG_EVERY steps
		WRITE_IMAGES : boolean
			Save images during logging
		LOSS_FUNC_STR : string
			Which loss function to use
		LOOP_AUTODIFF : string 
			How to save gradients through loop over timesteps. "checkpointed" or "lax"
		SPARSE_PRUNING : boolean
			Whether to prune model weights to a target sparsity
		TARGET_SPARSITY : float
			Target sparsity for model pruning - [0,1]
		key : jr.PRNGKey, optional
			Jax random number key. The default is jr.PRNGKey(int(time.time())).
		Returns
		-------
		None
		"""

		self.TRAIN_CONFIG = {
			"t":t,
			"iters":iters,
			"optimiser":optimiser,
			"STATE_REGULARISER":STATE_REGULARISER,
			"BOUNDARY_REGULARISER":BOUNDARY_REGULARISER,
			"WARMUP":WARMUP,
			"LOG_EVERY":LOG_EVERY,
			"CLEAR_CACHE_EVERY":CLEAR_CACHE_EVERY,
			"WRITE_IMAGES":WRITE_IMAGES,
			"LOSS_FUNC_STR":LOSS_FUNC_STR,
			"LOOP_AUTODIFF":LOOP_AUTODIFF,
			"SPARSE_PRUNING":SPARSE_PRUNING,
			"TARGET_SPARSITY":TARGET_SPARSITY
		}
		
		self.setup_logging("wandb",wandb_args=wandb_args)


		if LOSS_FUNC_STR=="l2":
			self._loss_func = loss.l2
		elif LOSS_FUNC_STR=="l1":
			self._loss_func = loss.l1
		elif LOSS_FUNC_STR=="vgg":
			self._loss_func = loss.vgg
		elif LOSS_FUNC_STR=="euclidean":
			self._loss_func = loss.euclidean
		elif LOSS_FUNC_STR=="spectral":
			self._loss_func = loss.spectral
		elif LOSS_FUNC_STR=="spectral_full":
			self._loss_func = loss.spectral_weighted
		elif LOSS_FUNC_STR=="rand_euclidean":
			#def _loss_func(self,x,y,dummy_key):
			#	return loss.random_sampled_euclidean(x,y,key)
			self._loss_func = lambda x,y,dummy_key:loss.random_sampled_euclidean(x,y,key=key)




		#@partial(eqx.filter_jit,donate="all-except-first")
		@eqx.filter_jit
		def make_step(nca,x,y,t,opt_state,key):
			"""
			

			Parameters
			----------
			nca : object callable - (float32 [N_CHANNELS,_,_],PRNGKey) -> (float32 [N_CHANNELS,_,_])
				the NCA object to train
			x : float32 array [BATCHES,N,CHANNELS,_,_]
				NCA state
			y : float32 array [BATCHES,N,OBS_CHANNELS,_,_]
				true data
			t : int
				number of NCA timesteps between x[N] and x[N+1]
			opt_state : optax.OptState
				internal state of self.OPTIMISER
			key : jr.PRNGKey, optional
				Jax random number key. 
				
			Returns
			-------
			nca : object callable - (float32 array [N_CHANNELS,_,_],PRNGKey) -> (float32 array [N_CHANNELS,_,_])
				the NCA object with updated parameters
			opt_state : optax.OptState
				internal state of self.OPTIMISER, updated in line with having done one update step
			loss_x : (float32, (float32 array [BATCHES,N,CHANNELS,_,_], float32 array [BATCHES,N]))
				tuple of (mean_loss, (x,losses)), where mean_loss and losses are returned for logging purposes,
				and x is the updated NCA state after t iterations

			"""
			
			@eqx.filter_value_and_grad(has_aux=True)
			def compute_loss(nca_diff, nca_static, x, y, t, key):
				_nca = eqx.combine(nca_diff, nca_static)
				v_nca = jax.vmap(_nca, in_axes=(0, None, 0), out_axes=0, axis_name="N")
				vv_nca = lambda x, callback, key_array: jax.tree_util.tree_map(
					lambda leaf, cb, ka: v_nca(leaf, cb, ka),
					x, callback, key_array
				)

				reg_log = jnp.zeros(len(x))
				boundary_reg_log = jnp.zeros(len(x))
				v_intermediate_reg = lambda z: jnp.array(
					jax.tree_util.tree_map(self.intermediate_reg, z)
				)

				# single‐step loss over batch
				_loss_func = lambda a, b, k: self.loss_func(a, b, k)
				v_loss_func = lambda a, b, ka: jnp.array(
					jax.tree_util.tree_map(_loss_func, a, b, ka)
				)

				def nca_step(carry, xs):
					j, dt_i = xs                   # dt_i: array shape [B]
					key, x, reg_log, boundary_reg_log = carry

					# RNG
					key = jr.fold_in(key, j)
					key_array = key_pytree_gen(key, (len(x), x[0].shape[0]))

					# one proposal
					x_proposed = vv_nca(x, self.BOUNDARY_CALLBACK, key_array)

					# elementwise delta
					delta = jax.tree_util.tree_map(lambda new, old: new - old,
												x_proposed, x)

					# **batch‐loop** scaling**
					x = [
						old + dt_i[b] * d
						for b, (old, d) in enumerate(zip(x, delta))
					]

					# regularisation
					reg_log         = reg_log + v_intermediate_reg(x)
					boundary_reg_log= boundary_reg_log + self.boundary_regulariser(x)

					return (key, x, reg_log, boundary_reg_log), None


				# Build the scan indices and dt‐steps
				# t == TIME_SAMPLING, number of micro‐steps per data interval
				indices = jnp.arange(t)                    # shape [t]
				# pick the correct interval k for this subtrajectory; here, 0 is a placeholder
				dt_interval = self.dt_seq[:, 0]            # shape [B]
				# we still want t micro‐steps, each of size dt_interval / t
				dt_per_micro = dt_interval / t             # shape [B]
				# broadcast to [t, B]
				dt_steps = jnp.broadcast_to(dt_per_micro[None, :], (t, dt_per_micro.shape[0]))
				xs = (indices, dt_steps)

				# run the scan
				(key, x, reg_log, boundary_reg_log), _ = eqx.internal.scan(
					nca_step,
					(key, x, reg_log, boundary_reg_log),
					xs=xs,
					length=t,
					kind=LOOP_AUTODIFF
				)

				# compute final loss against true y
				loss_key = key_pytree_gen(key, (len(x),))
				losses   = v_loss_func(x, y, loss_key)
				mean_loss = (jnp.mean(losses)
							+ STATE_REGULARISER * (jnp.mean(reg_log) / t)
							+ BOUNDARY_REGULARISER * (jnp.mean(boundary_reg_log) / t))
				return mean_loss, (x, losses)


			
			nca_diff,nca_static = nca.partition()
			loss_x,grads = compute_loss(nca_diff,nca_static,x,y,t,key)
			updates,opt_state = self.OPTIMISER.update(grads, opt_state, nca_diff)
			nca = eqx.apply_updates(nca,updates)
			(mean_loss,(x,losses)) = loss_x
			return nca,x,y,t,opt_state,key,mean_loss,losses
		
		nca = self.NCA_model
		nca_diff,nca_static = nca.partition()
		

		#--- OPTIMISER ---
		# Set up optimiser
		if optimiser is None:
			schedule = optax.exponential_decay(1e-3, transition_steps=iters, decay_rate=0.99)
			self.OPTIMISER = optax.nadam(schedule)
			
		else:
			self.OPTIMISER = optimiser
		opt_state = self.OPTIMISER.init(nca_diff)
		
		# # Split data into x and y
		x,y = self.DATA_AUGMENTER.data_load(key)
		
		
		best_loss = 100000000
		loss_thresh = 1e16
		model_saved = False
		loss_diff = 0
		#prev_loss = 0
		mean_loss = 0
		loss_diff_thresh = 1e-2
		error = 0
		error_at = 0
		SPARSITY = jnp.concat((jnp.zeros(WARMUP),jnp.linspace(0,TARGET_SPARSITY,iters-WARMUP)))

		pbar = tqdm(range(iters))
		#--- Do training run ---
		for i in pbar:
			#prev_loss = mean_loss
			if i%CLEAR_CACHE_EVERY==0:
				#print(f"Clearing cache at step {i}")
				jax.clear_caches()
			key = jr.fold_in(key,i)

			#nca,opt_state,(mean_loss,(x,losses)) = make_step(nca, x, y, t, opt_state,key)
			nca,x_new,y_new,t,opt_state,key,mean_loss,losses = make_step(nca, x, y, t, opt_state,key)
			loss_diff = mean_loss - best_loss


			pbar.set_postfix({'loss': mean_loss,'best loss': best_loss,'loss diff':loss_diff})

			if SPARSE_PRUNING:
				
				if i>WARMUP:

					ws,_ = nca.get_weights()
					sparsity_distribution = partial(jaxpruner.sparsity_distributions.uniform, sparsity=SPARSITY[i])
					pruner = jaxpruner.MagnitudePruning(
						sparsity_distribution_fn=sparsity_distribution,
						skip_gradients=True)
					ws = pruner.instant_sparsify(ws)[0]
					nca.set_weights(ws)

			
			if self.IS_LOGGING:
				self.LOGGER.tb_training_loop_log_sequence(losses, x_new, i, nca,write_images=WRITE_IMAGES,LOG_EVERY=LOG_EVERY)
			
			if jnp.isnan(mean_loss):
				error = 1
				error_at=i
				break
			elif any(list(map(lambda x: jnp.any(jnp.isnan(x)), x))):
				error = 2
				error_at=i
				break
			elif mean_loss>loss_thresh:
				error = 3
				error_at=i
				break
			
			# Do data augmentation update
			if error==0:
				#if i%UPDATE_DATA_EVERY==0 or i<WARMUP:
				if loss_diff<loss_diff_thresh or i<WARMUP:
					x,y = self.DATA_AUGMENTER.data_callback(x_new, y_new, i, key)
				
				
				# Save model whenever mean_loss beats the previous best loss
				if i>WARMUP:
					if mean_loss < best_loss:
						model_saved=True
						self.NCA_model = nca
						self.NCA_model.save(self.MODEL_PATH,overwrite=True)
						best_loss = mean_loss
						#tqdm.write("--- Model saved at "+str(i)+" epochs with loss "+str(mean_loss)+" ---")
		
		if error==0:
			print("Training completed successfully")
		elif error==1:
			print("|-|-|-|-|-|-  Loss reached NaN at step "+str(error_at)+" -|-|-|-|-|-|")
		elif error==2:
			print("|-|-|-|-|-|-  X reached NaN at step "+str(error_at)+" -|-|-|-|-|-|")
		elif error==3:
			print( "|-|-|-|-|-|-  Loss exceded "+str(loss_thresh)+" at step "+str(error_at)+", optimisation probably diverging  -|-|-|-|-|-|")
		if error!=0 and model_saved==False:
			print("|-|-|-|-|-|-  Training did not converge, model was not saved  -|-|-|-|-|-|")
		elif self.IS_LOGGING and model_saved:
			x,y = self.DATA_AUGMENTER.split_x_y(1)
			x,y = self.DATA_AUGMENTER.data_callback(x,y,0,key)
			#try:
			self.LOGGER.tb_training_end_log(self.NCA_model,x,t*x[0].shape[0],self.BOUNDARY_CALLBACK)
			# except Exception as e:
			# 	print("Error logging training end")
			# 	print(e)
			# 	pass