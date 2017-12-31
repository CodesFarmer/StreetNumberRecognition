import tensorflow as tf
from tensorflow.python.framework import graph_util
import os


def freeze_graph(in_model_dir, out_model_dir, net_name, output_nodes='', rename_outputs=None):
    #load checkpoint
    checkpoint = tf.train.get_checkpoint_state(in_model_dir)
    input_checkpoint = checkpoint.model_checkpoint_path
    #get the filename and path
    _, filename = os.path.split(input_checkpoint)
    file_prefix, _ = os.path.splitext(filename)
    out_graph = '%s/%s_%s.pb' % (out_model_dir, file_prefix, net_name)
    print(out_graph)

    #clear devices for load different graph without collision
    saver = tf.train.import_meta_graph(input_checkpoint + '.meta', clear_devices=True)
    in_graph = tf.get_default_graph()
    old_names = output_nodes.split(',')
    old_names = list(old_names)

    # #Rename the tensor within the neural network
    # if rename_outputs != None:
    #     new_names = rename_outputs.split(',')
    #     with in_graph.as_default():
    #         for o, n in zip(old_names, new_names):
    #             _out = tf.identity(in_graph.get_tensor_by_name(o + ':0'), name=n)
    #         old_names = new_names
    #
    in_graph_def = in_graph.as_graph_def()

    out_nodes = [n.name for n in in_graph_def.node if n.name.startswith(net_name)]
    out_nodes = [n for n in out_nodes if not 'random_normal' in n]
    out_nodes = [n for n in out_nodes if not 'truncated_normal' in n]
    out_nodes = [n for n in out_nodes if not 'Initializer/zeros' in n]
    out_nodes = [n for n in out_nodes if not 'Momentum' in n]
    out_nodes = [n for n in out_nodes if not 'Adam' in n]
    for nodes in out_nodes:
        print(nodes)
    with tf.Session(graph=in_graph) as sess:
        saver.restore(sess, input_checkpoint)
        out_graph_def = graph_util.convert_variables_to_constants(
            sess=sess, input_graph_def=in_graph_def, output_node_names=out_nodes
        )
        #write to disk
        with tf.gfile.GFile(out_graph, 'wb') as file:
            file.write(out_graph_def.SerializeToString())
        print('%d ops in final graph' % (len(out_graph_def.node)))


model_dir = 'model/model/model'
freeze_graph(model_dir, 'graph', 'pnet')
# #
# # load the model
# def load_freeze_model(sess, model_path):
#     with tf.gfile.GFile(model_path, 'rb') as file:
#         graph_def = tf.GraphDef()
#         graph_def.ParseFromString(file.read())
#         sess.graph.as_default()
#         tf.import_graph_def(graph_def, name='')
#
# print('Load successly!')
# constant_value = {}
# with tf.Session() as sess:
#     load_freeze_model(sess, 'graph/model_pnet.pb')
#     constant_ops = [op for op in sess.graph.get_operations() if op.type == 'Const']
#     for ctop in constant_ops:
#         constant_value[ctop.name] = sess.run(ctop.outputs[0])