import tensorflow as tf
import neuralnetwork.handnn as hnn

_WEIGHT_DECAY = 5e-5
_MOMENTUM = 0.9
_HARD_RATIO=0.6
_SIGMOID = 30
_NUM_HARD = 96

def weighted_loss(loss_elem):
    loss_elem = tf.convert_to_tensor(loss_elem)
    num = int(loss_elem.get_shape()[0])
    index = range(1, num+1)
    index = tf.convert_to_tensor(index)
    weights = tf.exp(tf.negative(index)/_SIGMOID)
    return tf.matmul(loss_elem, weights, transpose_b=True)

class pnet(hnn.handNN):
    def setup(self):
        (
            self.feed('data')
            .conv(3, 3, 1, 1, 10, name='conv1', padding='VALID')
            .activate(name='prelu1', activation='PReLU')
            .pool(2, 2, 2, 2, name='pool1', ptype_nn='MAX', padding='VALID')
            .conv(3, 3, 1, 1, 16, name='conv2', padding='VALID')
            .activate(name='prelu2', activation='PReLU')
            .conv(3, 3, 1, 1, 32, name='conv3', padding='VALID')
            .activate(name='prelu3', activation='PReLU')
            .conv(1, 1, 1, 1, 2, name='prob', padding='VALID')
            .softmax(3, name='prob_sm')
        )
        (   self.feed('prelu3')
            .conv(1, 1, 1, 1, 4, name='coor', padding='VALID')
        )

class rnet(hnn.handNN):
    def setup(self):
        (
            self.feed('data')
            .conv(3, 3, 1, 1, 28, name='conv1', padding='VALID')
            .activate(name='prelu1', activation='PReLU')
            .pool(3, 3, 2, 2, name='pool1', ptype_nn='MAX', padding='VALID')
            .conv(3, 3, 1, 1, 48, name='conv2', padding='VALID')
            .activate(name='prelu2', activation='PReLU')
            .pool(3, 3, 2, 2, name='pool2', ptype_nn='MAX', padding='VALID')
            .conv(2, 2, 1, 1, 64, name='conv3', padding='VALID')
            .activate(name='prelu3', activation='PReLU')
            .fc(128, name='fc1')
            .activate(name='prelu4', activation='PReLU')
            .fc(2, name='prob')
            .softmax(1, name='prob_sm')
        )
        (
            self.feed('prelu4')
            .fc(4, name='coor')
        )

class onet(hnn.handNN):
    def setup(self):
        (
            self.feed('data')
            .conv(3, 3, 1, 1, 32, name='conv1', padding='VALID')
            .activate(name='prelu1', activation='PReLU')
            .pool(3, 3, 2, 2, name='pool1', ptype_nn='MAX', padding='VALID')
            .conv(3, 3, 1, 1, 64, name='conv2', padding='VALID')
            .activate(name='prelu2', activation='PReLU')
            .pool(3, 3, 2, 2, name='pool2', ptype_nn='MAX', padding='VALID')
            .conv(3, 3, 1, 1, 64, name='conv3', padding='VALID')
            .activate(name='prelu3', activation='PReLU')
            .pool(2, 2, 2, 2, name='pool3', ptype_nn='MAX', padding='VALID')
            .conv(2, 2, 1, 1, 128, name='conv4', padding='VALID')
            .activate(name='prelu4', activation='PReLU')
            .fc(256, name='fc1')
            .activate(name='prelu5', activation='PReLU')
            .fc(2, name='prob')
            .softmax(1, name='prob_sm')
        )
        (
            self.feed('prelu5')
            .fc(4, name='coor')
        )

class onet_mn(hnn.handNN):
    def setup(self):
        (
            self.feed('data')
            .conv(3, 3, 1, 1, 32, name='conv1', padding='VALID')
            .activate(name='prelu1', activation='ReLU')
            .pool(3, 3, 2, 2, name='pool1', ptype_nn='MAX', padding='VALID')
            .mobile_unit(filters_num=64, name='mbn1', strdies=[1, 2, 2, 1], padding='VALID')
            .mobile_unit(filters_num=64, name='mbn2', strdies=[1, 2, 2, 1], padding='VALID')
            .pool(2, 2, 2, 2, name='pool2', ptype_nn='MAX', padding='VALID')
            # .mobile_unit(filters_num=64, name='mbn3', strdies=[1, 2, 2, 1], padding='VALID')
            # .pool(2, 2, 2, 2, name='pool2', ptype_nn='MAX', padding='VALID')
            .mobile_unit(filters_num=128, name='mbn3', strdies=[1, 1, 1, 1], padding='SAME')
            .fc(256, name='fc1')
            .activate(name='prelu5', activation='ReLU')
            .fc(2, name='prob')
            .softmax(1, name='prob_sm')
        )
        (
            self.feed('prelu5')
            .fc(4, name='coor')
        )

class onet_mn_sc(hnn.handNN):
    def setup(self):
        (
            self.feed('data')
            .conv(3, 3, 1, 1, 32, name='conv1', padding='VALID')
            .activate(name='prelu1', activation='ReLU')
            .pool(3, 3, 2, 2, name='pool1', ptype_nn='MAX', padding='VALID')
            .mobile_shortcut(filters_num=64, name='mbn1', strdies=[1, 2, 2, 1], padding='SAME')
            .mobile_shortcut(filters_num=64, name='mbn2', strdies=[1, 2, 2, 1], padding='SAME')
            .mobile_shortcut(filters_num=128, name='mbn3', strdies=[1, 1, 1, 1], padding='SAME')
            .fc(256, name='fc1')
            .activate(name='prelu5', activation='ReLU')
            .fc(2, name='prob')
            .softmax(1, name='prob_sm')
        )
        (
            self.feed('prelu5')
            .fc(4, name='coor')
        )

def model_nn_fn(features, labels, mode, params):
    # in_size = 12
    category, groundtruth = tf.split(labels, [1, 4], 1)
    #transpose the data from NCHW to NHWC
    inputs = tf.transpose(features['x'], [0, 2, 3, 1])
    #there are two part of output: probability and regression
    with tf.variable_scope(params['neuralnetwork'].lower()):
        if params['neuralnetwork'].lower() == 'pnet':
            neuralnetwork = pnet({'data': inputs})
        elif params['neuralnetwork'].lower() == 'rnet':
            neuralnetwork = rnet({'data': inputs})
        elif params['neuralnetwork'].lower() == 'onet':
            neuralnetwork = onet_mn({'data': inputs})
        else:
            raise KeyError('Unknow neural network type %s' % params['neuralnetwork'])
        neuralnetwork.is_training = mode == tf.estimator.ModeKeys.TRAIN
        neuralnetwork.setup()
    reg_weights = params['reg_weights']
    prob = neuralnetwork.layers['prob']
    coor = neuralnetwork.layers['coor']
    # define the loss, but we transfer the single label into matrix at first
    num_one = tf.constant(-1, tf.int32)
    num_zero = tf.constant(0, tf.int32)
    category = tf.cast(category, tf.int32)
    mask_cat = tf.not_equal(category, num_one)
    mask_reg = tf.not_equal(category, num_zero)
    category = tf.one_hot(category, 3)
    category = category[:, :, 0:2]
    category = tf.reshape(category, shape=[-1, 1, 1, 2])
    groundtruth = tf.reshape(groundtruth, shape=[-1, 1, 1, 4])
    if params['neuralnetwork'].lower() != 'pnet':
        # reshape the probability
        prob = tf.reshape(prob, shape=[-1, 1, 1, 2])
        coor = tf.reshape(coor, shape=[-1, 1, 1, 4])
    prob = tf.boolean_mask(prob, mask_cat)
    category = tf.boolean_mask(category, mask_cat)
    coor = tf.boolean_mask(coor, mask_reg)
    groundtruth = tf.boolean_mask(groundtruth, mask_reg)
    cross_entropy_all = tf.nn.softmax_cross_entropy_with_logits(labels=category, logits=prob)
    squared_error_all = tf.squared_difference(x=coor, y=groundtruth)
    squared_error_all = tf.reshape(squared_error_all, [-1])
    squared_error_all,_ = tf.nn.top_k(squared_error_all, k=_NUM_HARD, sorted=True)
    #weighted loss
    # index = [i for i in range(1, _NUM_HARD+1)]
    # index = tf.convert_to_tensor(index, dtype=tf.float32)
    # weights = tf.exp(tf.negative(index)/_SIGMOID)
    # squared_error = tf.reduce_sum(tf.multiply(squared_error_all, weights))
    # squared_error = weighted_loss(squared_error_all)
    cross_entropy = tf.reduce_mean(cross_entropy_all)
    squared_error = tf.reduce_mean(squared_error_all)
    weight_decay = _WEIGHT_DECAY*tf.add_n([tf.nn.l2_loss(var) for var in tf.trainable_variables()])
    loss = cross_entropy + reg_weights*squared_error + weight_decay
    #we set the loss for display
    tf.identity(cross_entropy, 'cross_entropy')
    tf.summary.scalar('cross_entropy', cross_entropy)
    tf.identity(squared_error, 'squared_error')
    tf.summary.scalar('squared_error', squared_error)
    if mode == tf.estimator.ModeKeys.TRAIN:
        #set the train_op
        global_steps = tf.train.get_or_create_global_step()
        # set the learning rate
        initial_learning_rate = 0.001
        # Multiply the learning rate by 0.1 at 100, 150, and 200 epochs.
        epochs = params['epochs']
        boundaries = [int(params['max_steps'] * epoch) for epoch in [(epochs*2)/5, (epochs*3)/5, (epochs*4)/5]]
        values = [initial_learning_rate * decay for decay in [1, 0.1, 0.01, 0.001]]
        learning_rate = tf.train.piecewise_constant(
            tf.cast(global_steps, tf.int32), boundaries, values)
        # Create a tensor named learning_rate for logging purposes
        tf.identity(learning_rate, name='learning_rate')
        tf.summary.scalar('learning_rate', learning_rate)
        optimizer = tf.train.MomentumOptimizer(
            learning_rate=learning_rate,
            momentum=_MOMENTUM
        )
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_op = optimizer.minimize(loss, global_step=global_steps)
    else:
        train_op = None
    predictions = {
        'classes': tf.argmax(prob, axis=2),
        'probabilities': tf.nn.softmax(prob, name='softmax_tensor')
    }
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)
    accuracy = tf.metrics.accuracy(
        tf.argmax(prob, axis=2),
        tf.argmax(category, axis=2)
    )
    metrics = {'accuracy': accuracy}
    #draw a curve for accuracy on test data
    tf.identity(accuracy[1], name='train_accuracy')
    tf.summary.scalar('train_accuracy', accuracy[1])

    return tf.estimator.EstimatorSpec(
        mode=mode,
        train_op=train_op,
        loss=loss,
        predictions=predictions,
        eval_metric_ops=metrics
    )

def create_pnet(sess, model_dir):
    with tf.variable_scope('pnet'):
        data = tf.placeholder(dtype=tf.float32, shape=[None, None, None, 1])
        neuralnetwork_p = pnet({'data': data})
        neuralnetwork_p.is_training = False
        neuralnetwork_p.setup()
        tf.train.Saver().restore(sess, model_dir)
        # neuralnetwork_p.save(sess, 'model/pnet.pb')
    pnet_fun = lambda img : sess.run((neuralnetwork_p.layers['coor'], neuralnetwork_p.layers['prob_sm']), feed_dict={data : img})
    return pnet_fun

def create_rnet(sess, model_dir):
    with tf.variable_scope('rnet'):
        data = tf.placeholder(dtype=tf.float32, shape=[None, 24, 24, 1])
        neuralnetwork_r = rnet({'data': data})
        neuralnetwork_r.is_training = False
        neuralnetwork_r.setup()
        tf.train.Saver().restore(sess, model_dir)
        # neuralnetwork_r.save(sess, 'model/rnet.pb')
    rnet_fun = lambda img : sess.run((neuralnetwork_r.layers['coor'], neuralnetwork_r.layers['prob_sm']), feed_dict={data : img})
    return rnet_fun

def create_onet(sess, model_dir):
    with tf.variable_scope('onet'):
        data = tf.placeholder(dtype=tf.float32, shape=[None, 48, 48, 1])
        neuralnetwork_o = onet_mn({'data': data})
        neuralnetwork_o.is_training = False
        neuralnetwork_o.setup()
        tf.train.Saver().restore(sess, model_dir)
    neuralnetwork_o.save(sess, 'model/onet_mn_wl75_rw3.pb')
    onet_fun = lambda img : sess.run((neuralnetwork_o.layers['coor'], neuralnetwork_o.layers['prob_sm']), feed_dict={data : img})
    return onet_fun


def create_dnn(sess, model_dir, params):
    # with tf.variable_scope('pnet'):
    with tf.variable_scope('pnet'):
        data1 = tf.placeholder(dtype=tf.float32, shape=[None, None, None, 1])
        neuralnetwork_p = pnet({'data': data1})
        neuralnetwork_p.is_training = False
        neuralnetwork_p.setup()
    with tf.variable_scope('rnet'):
        data2 = tf.placeholder(dtype=tf.float32, shape=[None, 24, 24, 1])
        neuralnetwork_r = rnet({'data': data2})
        neuralnetwork_r.is_training = False
        neuralnetwork_r.setup()
    with tf.variable_scope('onet'):
        data3 = tf.placeholder(dtype=tf.float32, shape=[None, 48, 48, 1])
        neuralnetwork_o = onet({'data': data3})
        neuralnetwork_o.is_training = False
        neuralnetwork_o.setup()
    all_vars = tf.all_variables()
    pnet_vars = [k for k in all_vars if k.name.startswith('pnet')]
    rnet_vars = [k for k in all_vars if k.name.startswith('rnet')]
    onet_vars = [k for k in all_vars if k.name.startswith('onet')]
    pnet_saver = tf.train.Saver(pnet_vars)
    rnet_saver = tf.train.Saver(rnet_vars)
    onet_saver = tf.train.Saver(onet_vars)
    pnet_saver.restore(sess, model_dir[0])
    rnet_saver.restore(sess, model_dir[1])
    onet_saver.restore(sess, model_dir[2])
    run_options = None
    run_metadata = None
    if 'options' in params:
        run_options = params['options']
    if 'metadata' in params:
        run_metadata = params['metadata']
    pnet_fun = lambda img : sess.run((neuralnetwork_p.layers['coor'], neuralnetwork_p.layers['prob_sm']),
                                     options=run_options, run_metadata=run_metadata, feed_dict={data1 : img})
    rnet_fun = lambda img : sess.run((neuralnetwork_r.layers['coor'], neuralnetwork_r.layers['prob_sm']),
                                     options=run_options, run_metadata=run_metadata, feed_dict={data2 : img})
    onet_fun = lambda img : sess.run((neuralnetwork_o.layers['coor'], neuralnetwork_o.layers['prob_sm']),
                                     options=run_options, run_metadata=run_metadata, feed_dict={data3 : img})
    # pnet_fun_pr = lambda img :print(sess.run((neuralnetwork_p.layers['prelu3'], 'prob/weight:0', 'prob/bias:0'), feed_dict={data : img}))
    return pnet_fun, rnet_fun, onet_fun

def load_dnn(sess, model_dir, params):
    # with tf.variable_scope('pnet'):
    with tf.variable_scope('pnet'):
        data1 = tf.placeholder(dtype=tf.float32, shape=[None, None, None, 1])
        neuralnetwork_p = pnet({'data': data1})
        neuralnetwork_p.is_training = False
        neuralnetwork_p.setup()
    neuralnetwork_p.load(sess, model_dir[0])
    with tf.variable_scope('rnet'):
        data2 = tf.placeholder(dtype=tf.float32, shape=[None, 24, 24, 1])
        neuralnetwork_r = rnet({'data': data2})
        neuralnetwork_r.is_training = False
        neuralnetwork_r.setup()
    neuralnetwork_r.load(sess, model_dir[1])
    with tf.variable_scope('onet'):
        data3 = tf.placeholder(dtype=tf.float32, shape=[None, 48, 48, 1])
        neuralnetwork_o = onet({'data': data3})
        neuralnetwork_o.is_training = False
        neuralnetwork_o.setup()
    neuralnetwork_o.load(sess, model_dir[2])
    run_options = None
    run_metadata = None
    if 'options' in params:
        run_options = params['options']
    if 'metadata' in params:
        run_metadata = params['metadata']
    pnet_fun = lambda img : sess.run((neuralnetwork_p.layers['coor'], neuralnetwork_p.layers['prob_sm']),
                                     options=run_options, run_metadata=run_metadata, feed_dict={data1 : img})
    rnet_fun = lambda img : sess.run((neuralnetwork_r.layers['coor'], neuralnetwork_r.layers['prob_sm']),
                                     options=run_options, run_metadata=run_metadata, feed_dict={data2 : img})
    onet_fun = lambda img : sess.run((neuralnetwork_o.layers['coor'], neuralnetwork_o.layers['prob_sm']),
                                     options=run_options, run_metadata=run_metadata, feed_dict={data3 : img})
    # pnet_fun_pr = lambda img :print(sess.run((neuralnetwork_p.layers['prelu3'], 'prob/weight:0', 'prob/bias:0'), feed_dict={data : img}))
    return pnet_fun, rnet_fun, onet_fun