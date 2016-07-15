# hdDeepLearningStudy
Code etc for Deep Learning Study Group

## June 27, July 11 2016 - Hacker Dojo   
Recurrent network for image generation - Deep Mind   
https://arxiv.org/pdf/1502.04623v2.pdf  
Background and some references cited  
http://blog.evjang.com/2016/06/understanding-and-implementing.html - blog w. code for VAE  
http://arxiv.org/pdf/1312.6114v10.pdf - Variational Auto Encoder  
https://jmetzen.github.io/2015-11-27/vae.html - tf code for variational auto-encoder  
https://www.youtube.com/watch?v=P78QYjWh5sM  
http://arxiv.org/pdf/1310.8499v2.pdf - deep autoregressive networks - deep mind  
https://arxiv.org/pdf/1401.4082.pdf  - stochastic backpropagation and approx inference - deep mind  
http://www.cs.toronto.edu/~fritz/absps/colt93.html - keep neural simple by minimizing descr length - hinton  
https://github.com/vivanov879/draw - code  


## June 20, 2016 - Penninsula   
Recurrent models of visual attention - Deep Mind   
https://papers.nips.cc/paper/5542-recurrent-models-of-visual-attention.pdf   

## June 23, 29 2016 - SF
http://arxiv.org/pdf/1410.5401v2.pdf - Neural Turing Machines - Graves et. al.  
https://arxiv.org/pdf/1605.06065v1.pdf - One Shot Learning - DeepMind  
http://www.shortscience.org/paper?bibtexKey=journals/corr/1605.06065 - Larochell comments on One-Shot paper
https://github.com/shawntan/neural-turing-machines - Code  
https://www.reddit.com/r/MachineLearning/comments/2xcyrl/i_am_j%C3%BCrgen_schmidhuber_ama/cp4ecce - schmidhuber's comments
http://www.thespermwhale.com/jaseweston/ram/papers/paper_16.pdf
http://snowedin.net/tmp/Hochreiter2001.pdf - 
Reviews:
http://icml.cc/2016/reviews/839.txt
Code
https://github.com/brendenlake/omniglot
https://github.com/tristandeleu/ntm-one-shot
https://github.com/MLWave/extremely-simple-one-shot-learning

## June 13, 2016 - TBD, Penninsula
Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning:  
http://arxiv.org/pdf/1602.07261v1.pdf  

## June 9, 2016 - Galvanize
Visualizing and Understanding RNN:  
https://arxiv.org/pdf/1506.02078v2.pdf  

## June 6, 2016 - Hacker Dojo
Google inception paper - origin of 1x1 convolution layers  
http://arxiv.org/pdf/1409.4842v1.pdf  

## June 2, May 26, 2016 - Galvanize

Image segmentation with deep encoder-decoder

https://arxiv.org/pdf/1511.00561.pdf

## May 23, 2016 - Hacker Dojo

Compressed networks, reducing flops by pruning

https://arxiv.org/pdf/1510.00149.pdf

http://arxiv.org/pdf/1602.07360v3.pdf

## May 16, 2016

Word2Vec meets LDA:

http://arxiv.org/pdf/1605.02019v1.pdf - Paper

https://twitter.com/chrisemoody - Chris Moody's twiter with links to slides etc.

http://qpleple.com/topic-coherence-to-evaluate-topic-models/ - writeup on topic coherence


## May 9, 2016

https://arxiv.org/pdf/1603.05027v2.pdf - Update on microsoft resnet - identity mapping

http://gitxiv.com/posts/MwSDm6A4wPG7TcuPZ/recurrent-batch-normalization - batch normalization w. RNN


## May 2, 2016

Go playing DQN - AlphaGo

https://gogameguru.com/i/2016/03/deepmind-mastering-go.pdf

https://m.youtube.com/watch?sns=em&v=pgX4JSv4J70 - video of slide presentation on paper

https://en.m.wikipedia.org/wiki/List_of_Go_games#Lee.27s_Broken_Ladder_Game - Handling "ladders" in alphgo

https://en.m.wikipedia.org/wiki/Ladder_(Go) - ladders in go

_____________________________________________________________________________________________________________________
## April 25, 2016 - Microsoft Resnet
The Paper

http://arxiv.org/pdf/1512.03385v1.pdf 

References:

http://arxiv.org/pdf/1603.05027v2.pdf - Identity mapping paper

Code:

https://keunwoochoi.wordpress.com/2016/03/09/residual-networks-implementation-on-keras/ - keras code

https://github.com/ry/tensorflow-resnet/blob/master/resnet.py - tensorflow code

https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/skflow/resnet.py
_________________________________________________________________________________________________________________
## April 18, 2016 - Batch Normalization  
The Paper  
https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf  
http://gitxiv.com/posts/MwSDm6A4wPG7TcuPZ/recurrent-batch-normalization - Batch Normalization for RNN  


___________________________________________________________________________________________________________
## April 11, 2016 - Atari Game Playing DQN
The Paper
https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf)

Related references:

This adds 'soft' and 'hard' attention and the 4 frames are replaced with an LSTM layer:

http://gitxiv.com/posts/NDepNSCBJtngkbAW6/deep-attention-recurrent-q-network

http://home.uchicago.edu/~arij/journalclub/papers/2015_Mnih_et_al.pdf - Nature Paper

http://www.nature.com/nature/journal/v518/n7540/full/nature14236.html - videos at the bottom of the page

http://llcao.net/cu-deeplearning15/presentation/DeepMindNature-preso-w-David-Silver-RL.pdf - David Silver's slides

http://www.cogsci.ucsd.edu/~ajyu/Teaching/Cogs118A_wi09/Class0226/dayan_watkins.pdf

http://www0.cs.ucl.ac.uk/staff/d.silver/web/Teaching.html - David Silver

Implementation Examples:

http://stackoverflow.com/questions/35394446/why-doesnt-my-deep-q-network-master-a-simple-gridworld-tensorflow-how-to-ev?rq=1

http://www.danielslater.net/2016/03/deep-q-learning-pong-with-tensorflow.html

__________________________________________________________________________________________________________
##  March 3, 2016 Gated Feedback RNN
The Paper

"Gated RNN" (http://arxiv.org/pdf/1502.02367v4.pdf

-Background Material

http://arxiv.org/pdf/1506.00019v4.pdf - Lipton's excellent review of RNN  
http://www.nehalemlabs.net/prototype/blog/2013/10/10/implementing-a-recurrent-neural-network-in-python/ - Discussion of RNN and theano code for Elman network - Tiago Ramalho  
http://deeplearning.cs.cmu.edu/pdfs/Hochreiter97_lstm.pdf - Hochreiter's original paper on LSTM  
https://www.youtube.com/watch?v=izGl1YSH_JA - Hinton video on LSTM 

-Skylar Payne's GF RNN code  
https://github.com/skylarbpayne/hdDeepLearningStudy/tree/master/tensorflow  

-Slides
https://docs.google.com/presentation/d/1d2keyJxRlDcD1LTl_zjS3i45xDIh2-QvPWU3Te29TuM/edit?usp=sharing  
https://github.com/eadsjr/GFRNNs-nest/tree/master/diagrams/diagrams_formula  

## Reviews  
http://www.computervisionblog.com/2016/06/deep-learning-trends-iclr-2016.html  
https://indico.io/blog/iclr-2016-takeaways/  
