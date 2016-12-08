# -*- coding: utf-8 -*-

import tensorflow as tf

with tf.Graph().as_default() as g:
    with g.name_scope("myscope") as scope: # 有了这个scope，下面的op的name都是类似myscope/Placeholder这样的前缀
        sess = tf.Session(target='', graph = g, config=None) # target表示要连接的tf执行引擎
        print "graph version:", g.version # 0
        a = tf.placeholder("float")
        print a.op # 输出整个operation信息，跟下面g.get_operations返回结果一样
        print "graph version:", g.version # 1
        b = tf.placeholder("float")
        print "graph version:", g.version # 2
        c = tf.placeholder("float")
        print "graph version:", g.version # 3
        y1 = tf.mul(a, b) # 也可以写成a * b
        print "graph version:", g.version # 4
        y2 = tf.mul(y1, c) # 也可以写成y1 * c
        print "graph version:", g.version # 5
        operations = g.get_operations()
        for (i, op) in enumerate(operations):
            print "============ operation", i+1, "==========="
            print op # 一个结构，包括：name、op、attr、input等,不同op不一样
        assert y1.graph is g
        assert sess.graph is g
        print "================ graph object address ================"
        print sess.graph
        print "================ graph define ================"
        print sess.graph_def
        print "================ sess str ================"
        print sess.sess_str
        print sess.run(y1, feed_dict={a: 3, b: 3}) # 9.0 feed_dictgraph中的元素和值的映射
        print sess.run(fetches=[b,y1], feed_dict={a: 3, b: 3}, options=None, run_metadata=None) # 传入的feches和返回值的shape相同
        print sess.run({'ret_name':y1}, feed_dict={a: 3, b: 3}) # {'ret_name': 9.0} 传入的feches和返回值的shape相同

        assert tf.get_default_session() is not sess
        with sess.as_default(): # 把sess作为默认的session，那么tf.get_default_session就是sess, 否则不是
            assert tf.get_default_session() is sess

        h = sess.partial_run_setup([y1, y2], [a, b, c]) # 分阶段运行，参数指明了feches和feed_dict列表
        res = sess.partial_run(h, y1, feed_dict={a: 3, b: 4}) # 12 运行第一阶段
        res = sess.partial_run(h, y2, feed_dict={c: res}) # 144.0 运行第二阶段，其中使用了第一阶段的执行结果
        print "partial_run res:", res
        sess.close()
