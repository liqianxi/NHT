import tensorflow as tf
#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
print("TensorFlow version:", tf.__version__)
tf.enable_eager_execution()

def exp_map(H):
        epsilon = 1e-6
        # H.shape: batch x n-1 x k
        n_minus_1 = int(H.shape[-2])
        zero_row = tf.zeros((1,n_minus_1))        # 1 x n-1
        I = tf.eye(n_minus_1)                     # n-1 x n-1
        zero_I = tf.concat((zero_row, I), axis=0) # n x n-1

        v_mat = tf.matmul(zero_I, H)                  # batch x n x k
        #print('v_mat', v_mat)
        # v_mat = v_mat + tf.eye(n_minus_1+1,1)
        # print('v_mat', v_mat)
        v_norm_vec = tf.linalg.norm(v_mat,axis=1)     # batch x k
        #print('v_norm_vec', v_norm_vec)
        v_norm_row_vec = tf.expand_dims(v_norm_vec,1) # batch x 1 x k
        #print('v_norm_row_vec', v_norm_row_vec)
        v_norm_diag = tf.linalg.diag(v_norm_vec)      # batch x k x k
        inv_v_norm_diag = tf.linalg.diag(1/(v_norm_vec+epsilon))
        sin_over_norm = tf.linalg.matmul(tf.math.sin(v_norm_diag),inv_v_norm_diag) # batch x k x k
        #print('sin_over_norm', sin_over_norm)

        e1 = tf.eye(n_minus_1+1,1) # n x 1
        #print(e1.shape)
        cos_term = tf.linalg.matmul(e1, tf.math.cos(v_norm_row_vec))
        sin_term = tf.linalg.matmul(v_mat, sin_over_norm)
        
        exp_mapped_vs = cos_term+sin_term
        #print(exp_mapped_vs.shape)
        #print('zI',zero_I.shape)
        #print(tf.linalg.norm(exp_mapped_vs,axis=1))
        #input('wait')
        return exp_mapped_vs


H = tf.random.uniform(shape=[1,6,1],minval=-10,maxval=10)

unit_sphere_output = []
for i in range(2000):
    H = tf.random.uniform(shape=[1,2,1],minval=-10,maxval=10)
    out = exp_map(H)
    unit_sphere_output.append(out.numpy().squeeze())

import matplotlib.pyplot as plt
[x,y,z] = list(zip(*unit_sphere_output))

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(x,y,z)
plt.show()