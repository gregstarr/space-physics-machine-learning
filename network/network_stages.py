def station_conv(mag_input):
    """extracts features from the magnetic field information
    
    mag_input: 4D Tensor - batch (1) x stations (?) x time (128) x component (3)
    """
    # conv - conv - pool
    c1_kernel = tf.get_variable("C1_kernel", shape=[1,5,3,32], dtype=tf.float32, trainable=True)
    C1 = tf.nn.conv2d(mag_input, c1_kernel, strides=[1,1,1,1], padding="SAME")
    A1 = tf.nn.leaky_relu(C1)
    
    c2_kernel = tf.get_variable("C2_kernel", shape=[1,3,32,32], dtype=tf.float32, trainable=True)
    C2 = tf.nn.conv2d(A1, c2_kernel, strides=[1,1,1,1], padding="SAME")
    A2 = tf.nn.leaky_relu(C2)
    
    P1 = tf.nn.max_pool(A2, ksize=[1,1,2,1], strides=[1,1,2,1], padding="SAME")
    # batch x stations x 64 x 32
    
    # conv - conv - pool
    c3_kernel = tf.get_variable("C3_kernel", shape=[1,3,32,64], dtype=tf.float32, trainable=True)
    C3 = tf.nn.conv2d(P1, c3_kernel, strides=[1,1,1,1], padding="SAME")
    A3 = tf.nn.leaky_relu(C3)
    
    c4_kernel = tf.get_variable("C4_kernel", shape=[1,3,64,64], dtype=tf.float32, trainable=True)
    C4 = tf.nn.conv2d(A3, c4_kernel, strides=[1,1,1,1], padding="SAME")
    A4 = tf.nn.leaky_relu(C4)
    
    P2 = tf.nn.max_pool(A4, ksize=[1,1,2,1], strides=[1,1,2,1], padding="SAME")
    # batch x stations x 32 x 64
    
    # conv - pool
    c5_kernel = tf.get_variable("C5_kernel", shape=[1,3,64,64], dtype=tf.float32, trainable=True)
    C5 = tf.nn.conv2d(P2, c5_kernel, strides=[1,1,1,1], padding="SAME")
    A5 = tf.nn.leaky_relu(C5)    
    mag_features = tf.nn.max_pool(A5, ksize=[1,1,2,1], strides=[1,1,2,1], padding="SAME")
    # batch x station x 16 x 64
    
    return mag_features


def station_net(mag_features, st_loc):
    """aggregates information from magnetic field data and positions
    
    mag_features: batch x stations x 16 x 64
    st_loc: stations x 3
    """
    
    # reshape and concatenate
    mag_sh = tf.shape(mag_features)
    station_vectors = tf.reshape(mag_features, [mag_sh[0], mag_sh[1], 1024])
    station_vectors = tf.concat([station_vectors, st_loc], axis=2)
    # batch x stations x 1027
    
    # FC net
    W1 = tf.get_variable("st_net_w1", shape=[1027, 1024], dtype=tf.float32, trainable=True)
    b1 = tf.get_variable("st_net_b1", shape=[1024], dtype=tf.float32, trainable=True)
    D1 = tf.reduce_sum(station_vectors[:,:,:,None] * W1[None,None,:,:], axis=2) + b1
    A1 = tf.nn.leaky_relu(D1)
    # batch x stations x 1024
    
    W2 = tf.get_variable("st_net_w2", shape=[1024, 1024], dtype=tf.float32, trainable=True)
    b2 = tf.get_variable("st_net_b2", shape=[1024], dtype=tf.float32, trainable=True)
    D2 = tf.reduce_sum(A1[:,:,:,None] * W2[None,None,:,:], axis=2) + b2
    st_features = tf.nn.leaky_relu(D2)
    # batch x stations x 1024
    
    return st_features


def global_conv(global_params):
    """extracts features from the global parameters (solar wind, etc.)
    """
    global_features = None
    return global_features


def aggregator(st_features, global_features=None):
    """adds up the station vectors, concatenates with the global features,
    outputs the substorm prediction
    
    st_features: 3D tensor - batch x stations x features
    """
    
    station_sum = tf.reduce_sum(st_features, axis=1)
    # batch x features
    
    # FC net
    D1 = tf.layers.dense(station_sum, 512, trainable=True)
    A1 = tf.nn.leaky_relu(D1)
    D2 = tf.layers.dense(A1, 512, trainable=True)
    A2 = tf.nn.leaky_relu(D2)
    model_output = tf.layers.dense(A2, 1, trainable=True) # yes/no, time, MLAT, sinMLT, cosMLT
    
    return model_output


def loss_function(model_output, ss_occurred, ss_time, ss_loc):
    """calculates loss based on model output and the substorm occurance
    
    model_output = batch x (ss_occurred, time, MLAT, sinMLT, cosMLT)
    """
    
    occurance_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=ss_occurred[:,0], logits=model_output[:,0])
    time_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=ss_time[:,0], logits=model_output[:,1])
    location_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=ss_loc, logits=model_output[:,2:])
    
    loss = tf.reduce_mean(occurance_loss)# + 0 * ss_occurred * (time_loss + location_loss))
    
    return loss