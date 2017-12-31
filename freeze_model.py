import tensorflow as tf

import neuralnetwork.handetector as hd

root_dir = '/home/slam/TestRoom/tensorflow_ep/handetect'

minsize = 12
factor = 0.709 # scale factor
model_path = []
# model_path.append('model/model_pnet.ckpt')
# model_path.append('model/model_rnet.ckpt')
# model_path.append('model/model_onet.ckpt')
model_path.append('%s/model/pnet.pb'%root_dir)
model_path.append('%s/model/rnet.pb'%root_dir)
# model_path.append('model/onet.pb')
# model_path.append('model/onet_quantized.pb')
# model_path.append('model/onet_mn_pool.pb')
# model_path.append('model/onet_mn_distilling_01_1_1_0_05_134.pb')
# model_path.append('model/onet_mn_distilling_pool.pb')
# model_path.append('model/onet_mn_sc_pool.pb')
model_path.append('%s/model/onet_mn_wl75_rw3.pb'%root_dir)
whichnet = 'onet'
whichop = 'mnwlr1'
# model_path.append('model/model_onetmn.ckpt')
threshold = [0.7, 0.6, 0.6]

sess = tf.Session()
run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
run_metadata = tf.RunMetadata()
params = {'options': run_options, 'metadata': run_metadata}
# proposal_pnet, refine_rnet, output_onet = hd.create_dnn(sess, model_path, params)
# proposal_pnet, refine_rnet, output_onet = hd.create_dn(sess, model_path)
# proposal_pnet = hd.create_pnet(sess, '%s/model/model/model_pnet.ckpt' % root_dir, '%s/graph/model_pnet.pb' % root_dir)
# proposal_pnet = hd.create_rnet(sess, '%s/model/model/model_rnet.ckpt' % root_dir, '%s/graph/model_rnet.pb' % root_dir)
# proposal_pnet = hd.create_rnet(sess, '%s/model/rnet_relu.pb' % root_dir, '%s/graph/model_rnet.pb' % root_dir)
proposal_pnet = hd.create_onet(sess, '%s/model/onet_mn_relu.pb' % root_dir, '%s/graph/model_onet.pb' % root_dir)
# proposal_pnet = hd.create_onet(sess, '%s/model/model/model_onet.ckpt' % root_dir, '%s/graph/model_onet.pb' % root_dir)
# rnet = hd.create_rnet(sess, model_path[1])
# onet = hd.create_onet(sess, 'model/model/model_onet.ckpt')