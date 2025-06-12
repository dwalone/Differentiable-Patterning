import jax.numpy as jnp
import jax.random as jr
import jax
import time
import equinox as eqx
from jax.experimental import mesh_utils
from Common.utils import key_pytree_gen
from Common.trainer.abstract_data_augmenter_tree import DataAugmenterAbstract
import itertools


class DataAugmenter(DataAugmenterAbstract):
   
    def __init__(self,data_true,hidden_channels=0, teacher_force_start=0.5, teacher_force_end=0.05,
                 teacher_force_decay_steps=2000):
        """
        Class for handling data augmentation for NCA training. 
        data_init is called before training,
        data_callback is called during training
        
        Also handles JAX array sharding, so all methods of NCA_trainer work
        on multi-gpu setups. Currently splits data onto different GPUs by batches


        Modified version of DataAugmenter where each batch can have different spatial resolution/size
        Treat data as Pytree of trajectories, where each leaf is a different batch f32[N,CHANNEL,WIDTH,HEIGHT]
        Parameters
        ----------
        data_true : PyTree [BATCHES] f32[N,CHANNELS,WIDTH,HEIGHT]
            true un-augmented data
        hidden_channels : int optional
            number of hidden channels to zero-pad to data. Defaults to zero
        """
        self.OBS_CHANNELS = data_true[0].shape[1]
        data_tree = []
        try:
            for i in range(data_true.shape[0]): # if data is provided as big array, convert to list of arrays. If data is list of arrays, this will leave it unchanged
                data_tree.append(data_true[i])
        except:
            data_tree = data_true
        data_true = jax.tree_util.tree_map(lambda x: jnp.pad(x,((0,0),(0,hidden_channels),(0,0),(0,0))),data_tree) # Pad zeros onto hidden channels


        self.data_true = data_true
        self.data_saved = data_true
        self.key = jax.random.PRNGKey(int(1000*time.time()))
        self.tf_start = teacher_force_start
        self.tf_end   = teacher_force_end
        self.tf_N     = teacher_force_decay_steps


    def data_callback(self,x,y,i,key=None):
        """
        Called after every training iteration to perform data augmentation and processing		


        Parameters
        ----------
        x : PyTree [BATCHES] f32[N-N_steps,CHANNELS,WIDTH,HEIGHT]
            Initial conditions
        y : PyTree [BATCHES] f32[N-N_steps,CHANNELS,WIDTH,HEIGHT]
            Final states
        i : int
            Current training iteration - useful for scheduling mid-training data augmentation

        Returns
        -------
        x : PyTree [BATCHES] f32[N-N_steps,CHANNELS,WIDTH,HEIGHT]
            Initial conditions
        y : PyTree [BATCHES] f32[N-N_steps,CHANNELS,WIDTH,HEIGHT]
            Final states

        """
        # tf_prob = jnp.clip(
        #     self.tf_start - (i / self.tf_N) * (self.tf_start - self.tf_end),
        #     self.tf_end, self.tf_start,
        # )
        # x_true, _ = self.split_x_y(1)
        # x = jittable_callback_bit(x, x_true, self.OBS_CHANNELS, i, tf_prob)
        # x = self.noise(x, 0.001, key=self.key)
        # self.key = jax.random.fold_in(self.key, i)
        # return x, y

        # ---------- PATCH 4: random roll-out horizons --------------------------
        step_ranges = [(5, 10), (15, 25), (30, 40)]   # like the reference code
        idx        = (i // 400) % len(step_ranges)    # cycle every 400 iters
        lo, hi     = step_ranges[idx]
        n_steps    = jr.randint(self.key, (), lo, hi + 1)

        # shift trajectories by a random number of steps
        propagate = lambda a: a.at[n_steps:].set(a[:-n_steps])
        x = jax.tree_util.tree_map(propagate, x)

        tf_prob = jnp.clip(
            self.tf_start - (i / self.tf_N) * (self.tf_start - self.tf_end),
            self.tf_end, self.tf_start,
        )
        x_true, _ = self.split_x_y(1)
        x = jittable_callback_bit(x, x_true, self.OBS_CHANNELS, i, tf_prob)

        # tiny input noise (kept from your original code)
        x = self.noise(x, 0.001, key=self.key)
        self.key = jax.random.fold_in(self.key, i)
        return x, y
        # ----------------------------------------------------------------------
    
@eqx.filter_jit
def jittable_callback_bit(x, x_true, obs_channels, step, tf_prob):
    # 1) shift trajectories one step forward
    propagate = lambda a: a.at[1:].set(a[:-1])
    x = jax.tree_util.tree_map(propagate, x)

    # 2) choose which batches to teacher-force
    key     = jax.random.PRNGKey(step)
    tf_mask = jax.random.bernoulli(key, p=tf_prob,
                                   shape=(len(x), 1, 1, 1, 1))
    tf_mask = list(tf_mask)        # match the pytree structure of x

    def maybe_reset(a, b, m):
        m = m.squeeze()            # scalar bool
        new0 = jnp.where(m,
                         b[0, :obs_channels],
                         a[0, :obs_channels])
        return a.at[0, :obs_channels].set(new0)

    x = jax.tree_util.tree_map(maybe_reset, x, x_true, tf_mask)
    return x


