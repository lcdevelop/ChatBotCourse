# lstm
A basic lstm network can be written from scratch in a few hundred lines of python, yet most of us have a hard time figuring out how lstm's actually work.  The original Neural Computation [paper](https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=3&cad=rja&uact=8&ved=0CDAQFjACahUKEwj1iZLX5efGAhVMpIgKHbv3DiI&url=http%3A%2F%2Fdeeplearning.cs.cmu.edu%2Fpdfs%2FHochreiter97_lstm.pdf&ei=ZuirVfW-GMzIogS777uQAg&usg=AFQjCNGoFvqrva4rDCNIcqNe_SiPL_VPxg&sig2=ZYnsGpdfHjRbK8xdr1thBg&bvm=bv.98197061,d.cGU) is too technical for non experts.  Most blogs online on the topic seem to be written by people
who have never implemented lstm's for people who will not implement them either.  Other blogs are written by experts (like this [blog post](http://karpathy.github.io/2015/05/21/rnn-effectiveness/)) and lack simplified illustrative source code that actually does something.  The [Apollo](https://github.com/Russell91/apollo) library built on top of caffe is terrific and features a fast lstm implementation.  However, the downside of efficient implementations is that the source code is hard to follow.

This repo features a minimal lstm implementation for people that are curious about lstms to the point of wanting to know how lstm's might be implemented.  The code here follows notational conventions set forth in [this](http://arxiv.org/abs/1506.00019)
well written tutorial introduction.  This article should be read before trying to understand this code (at least the part about lstm's).  By running `python test.py` you will have a minimal example of an lstm network learning to predict an output sequence of numbers in [-1,1] by using a Euclidean loss on the first element of each node's hidden layer.  

Play with code, add functionality, and try it on different datasets.  Pull requests welcome. 

Please read [my blog article](http://nicodjimenez.github.io/2014/08/08/lstm.html) if you want details on the backprop part of the code.

Also, check out a version of this code written in the D programming language by Mathias Baumann: https://github.com/Marenz/lstm
