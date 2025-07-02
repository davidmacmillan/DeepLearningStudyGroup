# Deep Learning Study Group  
Papers, code, etc. for the Deep Learning Study Group.   
Meeting time - Tuesdays, 6:30 pm California time on Zoom     
Zoom and Discord links are on the meetup page:      
https://www.meetup.com/handsonprogrammingevents/      

___________________________________________________________________________________________________________
# ======== 2025 ========   
#### Two blogs and a paper for July 8, 2025:   
Blog #1 - Gemma 3n model overview   
https://ai.google.dev/gemma/docs/gemma-3n   
Blog #2 - Introducing Gemma 3n: The developer guide   
https://developers.googleblog.com/en/introducing-gemma-3n-developer-guide/   
MatFormer: Nested Transformer for Elastic Inference   
https://arxiv.org/pdf/2310.07707   
There are multiple YouTubes on Gemma 3n and MatFormer.   
   
### Paper for July 1, 2025:   
MELODI: Exploring Memory Compression for Long Contexts (DeepMind, Oct. 2024)   
https://arxiv.org/abs/2410.03156   
Open Review:   
https://openreview.net/forum?id=TvGPP8i18S   

### Paper for June 24, 2025:   
Relaxed Recursive Transformers: Effective Parameter Sharing with Layer-wise LoRA   
https://arxiv.org/pdf/2410.20672   
OpenReview:   
https://openreview.net/forum?id=WwpYSOkkCt   
   
### Paper for June 17, 2025:   
Concise Reasoning via Reinforcement Learning    
https://arxiv.org/pdf/2504.05185   
    
### For June 10, 2025:   
Good news - no homework this week!!!   
At the meeting, one of our members, Ted, will present MultiDecode,    
original work he has done on speeding inference, including for RAG.   
   
### Papers for June 3, 2025:   
Efficient Sequence Transduction by Jointly Predicting Tokens and Durations    
https://arxiv.org/abs/2304.06795    
Fast Conformer with Linearly Scalable Attention for Efficient Speech Recognition    
https://arxiv.org/abs/2305.05084    
   
### Paper for May 27, 2025    
AlphaEvolve: A coding agent for scientific and algorithmic discovery   
https://storage.googleapis.com/deepmind-media/DeepMind.com/Blog/alphaevolve-a-gemini-powered-coding-agent-for-designing-advanced-algorithms/AlphaEvolve.pdf   
Blog:   
https://deepmind.google/discover/blog/alphaevolve-a-gemini-powered-coding-agent-for-designing-advanced-algorithms/   
   
### Paper for May 20, 2023   
Qwen3 Technical Report   
https://github.com/QwenLM/Qwen3/blob/main/Qwen3_Technical_Report.pdf   
There are many YouTubes.  
Also try it out (e.g. Ollama has it) or here:  
https://qwen3.app/   

### Paper for May 13, 2025:   
Flow matching for Generative Modeling   
https://arxiv.org/abs/2210.02747   
YouTube by Yannic Kilcher:   
https://youtu.be/7NNxK3CqaDk   
YouTube by Jia-Bin Huang (Univ. Maryland):   
https://youtu.be/DDq_pIfHqLs   
YouTube by Peter Abbeel (UC Berkeley):   
https://www.youtube.com/watch?v=SkSDCzz41Vs   
There are also other YouTubes and blogs such as:  
https://www.youtube.com/watch?v=7cMzfkWFWhI   
   
### Paper for May 6, 2025, from DeepMind:   
Round and Round We Go! What makes Rotary Positional Encodings useful?   
https://arxiv.org/pdf/2410.06205   
There is a YouTube from Gabriel Mongaras:   
https://www.youtube.com/watch?v=2tS_bXPoriI   
   
### Paper for April 29, 2025:
Why do LLMs attend to the first token?  
https://arxiv.org/abs/2504.02732  
As background, Evan Miller has a blog from 2023 on this issue and identified a simple fix:   
add +1 in the transformer softmax denominators (but not to the final LLM output softmax).  
https://www.evanmiller.org/attention-is-off-by-one.html    
Tracing the heritage, tonight's paper on pg. 3 references Xiao 2024  
https://arxiv.org/abs/2309.17453   
and Xiao (pg. 4 & 6) notes his StreamingLLM approach for attention sinks can  
(perhaps) be eliminated if one instead uses Miller's +1 softmax recommendation.  
Yannic Kilcher has a YouTube on Xaio:   
https://www.youtube.com/watch?v=409tNlaByds   
 
### For April 22, 2025 we will discuss Anthropic's MCP and Google's Agent2Agent.
Anthropic MCP   
https://www.anthropic.com/news/model-context-protocol   
MCP Introduction,  Tutorials, Concepts:   
https://modelcontextprotocol.io/introduction   
Google agent2agent    
https://developers.googleblog.com/en/a2a-a-new-era-of-agent-interoperability/   
A2A Technical Documentation:   
https://google.github.io/A2A/#/documentation   
A2A and MCP:   
https://google.github.io/A2A/#/topics/a2a_and_mcp   
   
### Paper (actually a blog) for April 15, 2025:
We are continuing the discussion from last week on the recent Anthropic papers/blogs.   
We are doing the second paper/blog this week:   
On the Biology of a Large Language Model     
https://transformer-circuits.pub/2025/attribution-graphs/biology.html    
Yannic Kilcher has a YouTube (part 1 of 2 parts is out so far):   
https://www.youtube.com/watch?v=mU3g2YPKlsA   
Sabine Hossenfelder has a YouTube:   
https://www.youtube.com/watch?v=-wzOetb-D3w  

### Paper (actually a blog) for April 8, 2025:  
Circuit Tracing: Revealing Computational Graphs in Language Models    
https://transformer-circuits.pub/2025/attribution-graphs/methods.html   
If you prefer reading a PDF version, try: https://webtopdf.com/   
Additional background reading:   
Faith and Fate: Limits of Transformers on Compositionality   
https://arxiv.org/abs/2305.18654   
On Limitations of the Transformer Architecture,  Chapter 3 - The Impossibility of Composition   
https://arxiv.org/abs/2402.08164   
   
### Paper for April 1, 2025:      
Fractal Generative Models    
https://arxiv.org/pdf/2502.17437    
YouTube:    
https://www.youtube.com/watch?v=yxNuUg3aUjA    
Github:    
https://github.com/LTH14/fractalgen    

### Paper for March 25, 2025:
From superposition to sparse codes: interpretable representations in neural networks   
https://arxiv.org/pdf/2503.01824   
There is at least one YouTube:   
https://www.youtube.com/watch?v=t_i2NRr2eZA   
   
### Paper for March 18, 2025:   
u-µP: The Unit-Scaled Maximal Update Parametrization   
https://arxiv.org/pdf/2407.17465   

### Paper for March 11, 2025 is a blog:   
The Ultra-Scale Playbook: Training LLMs on GPU Clusters   
https://huggingface.co/spaces/nanotron/ultrascale-playbook   
There are a number of YouTubes on this.   
   
### Paper for March 4, 2025:   
MONA: Myopic Optimization with Non-myopic Approval Can Mitigate Multi-step Reward Hacking (Deepmind)   
https://arxiv.org/pdf/2501.13011   
There is a blog:   
https://deepmindsafetyresearch.medium.com/mona-a-method-for-addressing-multi-step-reward-hacking-a31ac4b16483   
There is at least one YouTube:   
https://www.youtube.com/watch?v=mwqgIF3Ey8k   

### Paper for February 25, 2025:    
Scaling up Test-Time Compute with Latent Reasoning: A Recurrent Depth Approach    
https://arxiv.org/pdf/2502.05171    

### Paper for February 18, 2025:   
s1: Simple test-time scaling   
https://arxiv.org/abs/2501.19393   
Github:   
https://github.com/simplescaling/s1   
There are many YouTubes, including:   
https://www.youtube.com/watch?v=3tM3yc9UI84   
and that YouTube mentions three similar papers published on almost the same date as the S1 paper:   
Think Smarter not Harder: Adaptive Reasoning with Inference Aware Optimization (by Meta)   
https://arxiv.org/abs/2501.17974   
Large Language Models Think Too Fast To Explore Effectively (Georgia Institute of Tech.)   
https://arxiv.org/pdf/2501.18009   
Thoughts Are All Over the Place: On the Underthinking of o1-Like LLMs (TenCent AI Labs)   
https://arxiv.org/pdf/2501.18585   
   
### Paper for February 11, 2025:    
Large Concept Models: Language Modeling in a Sentence Representation Space    
https://ai.meta.com/research/publications/large-concept-models-language-modeling-in-a-sentence-representation-space/    
A YouTube (many others):    
https://www.youtube.com/watch?v=TwLiNTYvpPo    

### Paper for February 4, 2025:   
Deepseek R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning   
https://github.com/deepseek-ai/DeepSeek-R1/blob/main/DeepSeek_R1.pdf   
There are many YouTubes and lots of press coverage   
   
Base tech for R1 / background info / may also discuss if time:   
Deepseek-V3 Technical Report   
https://github.com/deepseek-ai/DeepSeek-V3/blob/main/DeepSeek_V3.pdf   
   
Other R1-related info:   
Berkeley Researchers Replicate Deepseek R1's Core Tech for Just $30   
https://xyzlabs.substack.com/p/berkeley-researchers-replicate-deepseek   
Jiayi Pan's discussion of what he and his team did:   
https://x.com/jiayi_pirate/status/1882839370505621655   
Berkeley team's code:  
https://github.com/Jiayi-Pan/TinyZero   

### Paper for January 28, 2025:   
READ THIS FIRST:   
This is a long paper (68 pages).    
They are doing some cool, non-standard stuff with transformers.    
That will be the focus of our discussion.    
The "assigned reading" is Architecture, pages 21-37 (first part of Appendix), including Algorithms & Figures.   
Skim the rest of the paper, as needed, to understand their context / what they are trying to do.   
We may also look at their GitHub code, so you may want to take a look at that also.   
\-\-\-   
Paper (focus on pages 21-37 - see the READ THIS above):  
Simulating 500 million years of evolution with a language model   
https://www.biorxiv.org/content/10.1101/2024.07.01.600583v2  <-- note v2 at end of URL   
Github for model (open source):   
https://github.com/evolutionaryscale/esm   
YouTube by paper author:   
https://www.youtube.com/watch?v=qeqbm8a1-ZA   
Project page:    
ESM3: Simulating 500 million years of evolution with a language model   
https://www.evolutionaryscale.ai/blog/esm3-release   
Huggingface for weights (open source license for non-commercial use; commercial use requires license):   
https://huggingface.co/EvolutionaryScale/esm3   
   
### Paper for January 21, 2025:   
Nash Learning from Human Feedback   
https://arxiv.org/pdf/2312.00886   
   
### Paper for January 14, 2025:
TokenFormer: Rethinking Transformer Scaling with Tokenized Model Parameters   
https://arxiv.org/pdf/2410.23168   
There are many YouTubes including by Yannic Kilcher:   
https://www.youtube.com/watch?v=gfU5y7qCxF0   
and Gabriel Mongaras:   
https://www.youtube.com/watch?v=4lGgbkD6Z0I   
   
### Paper for January 7, 2025   
Visual Autoregressive Modeling: Scalable Image Generation via Next-Scale Prediction   
https://arxiv.org/pdf/2404.02905   
There is at least one YouTube:   
https://www.youtube.com/watch?v=yJ396Ksiv2s   

# ======== 2024 ========  
### No paper or meeting for December 31, 2024 - Happy New Year!   
   
### No paper or meeting for December 24, 2024 - Happy Holidays!   
   
### Paper for Dec. 17, 2024:
Generative Reward Models     
https://arxiv.org/abs/2410.12832    
    
### Paper for Dec. 10, 2024:   
Grokked Transformers are Implicit Reasoners: A Mechanistic Journey to the Edge of Generalization   
https://arxiv.org/pdf/2405.15071   
Github:   
https://github.com/OSU-NLP-Group/GrokkedTransformer   
OpenReview:   
https://openreview.net/forum?id=ns8IH5Sn5y   
   
### Paper for December 3, 2024:   
Enhancing LLM Reasoning with Reward-guided Tree Search   
https://arxiv.org/abs/2411.11694   

### Paper for November 26, 2024:   
Discrete Diffusion Modeling by Estimating the Ratios of the Data Distribution   
https://arxiv.org/abs/2310.16834   
YouTube (shorter):   
https://www.youtube.com/watch?v=K_9wQ6LZNpI   
YouTube (longer, by primary paper author):   
https://www.youtube.com/watch?v=_1qv_LNjH9U   
Github:   
https://github.com/louaaron/Score-Entropy-Discrete-Diffusion   
   
### Paper for November 19, 2024:   
We will continue the discussion of:   
The Llama 3 Herd of Models   
https://arxiv.org/abs/2407.21783   
We will start the discussion with a focus on Sections 7 and 8 (which we didn't have time for last week).    
If time permits (it likely will) we will discuss (this week's new reading "assignment"):   
* Section 3.3 through end of Section 3.3.4 (~pages 8 - 14)   
* Section 6 (all of it) (~pages 51 - 53)   
* Section 5 (skim for what whatever results catch your interest) (~pages 28 - 51)   
* Any Figures and Tables that are referenced in the above readings.   
* Anything anywhere in the paper that you want to discuss.
   
There are multiple YouTubes on the paper.   
   
### Paper for November 12, 2024:   
The Llama 3 Herd of Models   
https://arxiv.org/abs/2407.21783   
This is a long paper (92 pg) so we are skipping the sections on hardware, inference and results (leaves ~30 pg to read).   
Our focus is on the software and architecture, including multi-modal aspects (the "assignment").   
At the meetup we will discuss the paper, not read through it. Bring your questions, comments, etc.   
Anyone is welcome to attend and listen without reading the "assignment".   
If nobody reads it, the meeting will be short.   
On the other hand, feel free to read more than the "assignment" and to share your wider insights in the meeting!   
Here is the "assigned" reading with precise Sections shown:   
* From the start through end of Section 3.2.1 (~pages 1 - 8)   
* Section 3.4 through end of Section 4.3.7 (~pages 14-28)   
* Section 7 through end of Section 7.5.7 (~pages 54-61)   
* Section 8 through end of Section 8.3.2 (~pages 63-66)   
* Any Figures and Tables that are referenced in the above readings.   

A copy of the paper with the above sections marked is in this Github here:    
https://github.com/davidmacmillan/DeepLearningStudyGroup/blob/master/The%20Llama%203%20Herd%20of%20Models%202407.21783v2.pdf   
There are multiple YouTubes on the paper.   

### Paper for November 5, 2024:   
Agent S: An Open Agentic Framework that Uses Computers Like a Human   
https://arxiv.org/pdf/2410.08164v1   
Github:   
https://github.com/simular-ai/Agent-S   
There are a number of YouTubes on this paper.   
   
### Paper for October 29, 2024:   
nGPT: Normalized Transformer with Representation Learning on the Hypersphere   
https://arxiv.org/pdf/2410.01131   
There appear to be multiple YouTubes.   
   
### Paper for October 22, 2024:   
Open discussion of AI coding assist & AI coding completion tools people have used, and their assessment of them.   
Interested in the full range of people's experiences with AI code tools: for code creation, code completion (copiloting), code debugging, and code refactoring.    
Examples of code welcome but not required.    

### Paper for October 15, 2024:   
Diffusion Models are Evolutionary Algorithms   
https://arxiv.org/pdf/2410.02543   
Tweet:   
https://x.com/YanboZhang3/status/1843134007892176995   
Github:   
https://github.com/Zhangyanbo/diffusion-evolution   
At least one YouTube:   
https://www.youtube.com/watch?v=Dh9gtg6N79U   
   
### Paper for October 8, 2024     
Scaling Scaling Laws with Board Games    
https://arxiv.org/pdf/2104.03113    
     
### Paper for October 1, 2024:   
Graph Retrieval-Augmented Generation: A Survey   
https://arxiv.org/abs/2408.08921    
YouTube (in Mandarin) (but click CC, then the Gear, then subtitles, then English):     
https://www.youtube.com/watch?v=1OsVlbhMkek    
   
### For September 24, 2024:    
Writing in the Margins: Better Inference Pattern for Long Context Retrieval   
https://www.arxiv.org/abs/2408.14906   
   
### Paper for September 17, 2024   
Diffusion Models Learn Low-Dimensional Distributions via Subspace Clustering∗   
https://www.arxiv.org/abs/2409.02426    
   
### Paper for September 10, 2024    
Paper for September 10, 2024    
Unexpected Benefits of Self-Modeling in Neural Systems    
https://arxiv.org/pdf/2407.10188    
YouTube video    
https://www.youtube.com/watch?v=yvHZ0nk8O5I    
    
### Paper for September 3, 2024
We are continuing the discussion of the paper from August 20, 2024:   
The Remarkable Robustness of LLMs: Stages of Inference?    
https://arxiv.org/abs/2406.19384    

### Paper for August 27, 2024:
*** This is a long paper!  Focus on pages 1-21 and skim Appendix D.8 as a representative output. ***    
The AI Scientist: Towards Fully Automated Open-Ended Scientific Discovery   
https://arxiv.org/pdf/2408.06292   
Enticing or disturbing tweet:   
https://x.com/Simeon_Cps/status/1823207094318735527   
Blog:      
https://sakana.ai/ai-scientist/   
Github:   
https://github.com/SakanaAI/AI-Scientist    

### Paper for August 20, 2024:   
The Remarkable Robustness of LLMs: Stages of Inference?    
https://arxiv.org/abs/2406.19384    
    
### Paper for August 13, 2024:   
Segment 2 Anything (arXiv version):    
https://arxiv.org/abs/2408.00714   
Additional resources - Meta's Blog:   
https://ai.meta.com/sam2/   
Meta's Interactive Demo:   
https://sam2.metademolab.com/   
Meta's Announcement:   
https://ai.meta.com/research/publications/sam-2-segment-anything-in-images-and-videos/   
Github:   
https://github.com/facebookresearch/segment-anything-2     
There are a number of YouTube videos.   

### Paper for August 6, 2024   
TextGrad: Automatic "Differentiation" via Text  
https://arxiv.org/abs/2406.07496    
Github:   
https://github.com/zou-group/textgrad   
Many YouTubes including:   
https://youtu.be/Qks4UEsRwl0   
   
### Paper for July 30, 2024   
The paper for July 30, 2024 is:   
DETRs Beat YOLOs on Real-time Object Detection   
https://arxiv.org/abs/2304.08069   
Additional Background Materials - Project page:   
https://zhao-yian.github.io/RTDETR/   
Video (demo only):   
https://www.youtube.com/watch?v=TbaLWroPYbo   
Github:   
https://github.com/lyuwenyu/RT-DETR   
Background video on normal DETR by Meta (creator also has videos on other object detection models):   
https://www.youtube.com/watch?v=A2f4w54fSsM   
   
### Paper for July 23, 2024   
When Can LLMs Actually Correct Their Own Mistakes? A Critical Survey of Self-Correction of LLMs   
https://arxiv.org/abs/2406.01297   

### Paper for July 16, 2024
xLSTM: Extended Long Short-Term Memory   
https://arxiv.org/abs/2405.04517   
YouTube (Yannic Kilcher)   
https://www.youtube.com/watch?v=0OaEv1a5jUM   
YouTube (Gabriel Mongaras)   
https://www.youtube.com/watch?v=4ND8lU2aN_k   
Medium article   
https://medium.com/@AIBites/xlstm-extended-long-short-term-memory-networks-c4ba34fdd98d   

### For July 9, 2024:   
6:30 PM - face-to-face get together & casual sit-down dinner. No paper this week.   
At: Agave Mexican Bistro, 194 Castro Street, Mountain View, California 94041.      

### Paper for July 2, 2024:
Paper:  Banishing LLM Hallucinations Requires Rethinking Generalization   
https://arxiv.org/abs/2406.17642   
Github:   
https://github.com/lamini-ai/   

### Paper for June 25, 2024:    
Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention    
https://arxiv.org/abs/2006.16236    
YouTube by Yannic Kilcher on  paper (may be others):    
https://www.youtube.com/watch?v=hAooAOFRsYc    

### Paper for June 18, 2024:   
Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality (Mamba 2)   
https://arxiv.org/abs/2405.21060   
This blog:  
https://gonzoml.substack.com/p/mamba-2-is-here   
and the 4 referenced blogs starting here:   
https://goombalab.github.io/blog/2024/mamba2-part1-model/   
are more approachable.   
   
### For June 11, 2024, will continue with the paper (blog):   
"Scaling Monosemanticity: Extracting Interpretable Features from Claude 3 Sonnet"   
https://transformer-circuits.pub/2024/scaling-monosemanticity/index.html   
Tweet thread / overview & highlights:   
https://x.com/mlpowered/status/1792948212728524917   

### Our paper for June 4, 2024 is a blog:   
"Scaling Monosemanticity: Extracting Interpretable Features from Claude 3 Sonnet"   
https://transformer-circuits.pub/2024/scaling-monosemanticity/index.html   
Tweet thread / overview & highlights:   
https://x.com/mlpowered/status/1792948212728524917   
Good video on this week's paper (blog):   
https://www.youtube.com/watch?v=y0ZXFl3rQlQ   

### Paper for May 28, 2024   
The Platonic Representation Hypothesis   
https://arxiv.org/pdf/2405.07987   
Github / project page:   
https://phillipi.github.io/prh/   
Github with their code:   
https://github.com/minyoungg/platonic-rep   
There are also a number of YouTubes that discuss the paper.   

### For May 21, 2024:   
Instead of a paper, we are going to go through Andrej Karpathy's YouTube video on creating transformer code:   
https://www.youtube.com/watch?v=kCc8FmEb1nY   
Have the colab or github code loaded on your PC before, ready to go through, so you don't have to type it in during our session.   
Colab:   
https://colab.research.google.com/drive/1JMLa53HDuA-i7ZBmqV7ZnA3c_fvtXnx-   
Github:   
https://github.com/karpathy/ng-video-lecture   

### Paper for May 14, 2024:   
KAN: Kolmogorov-Arnold Networks   
https://arxiv.org/pdf/2404.19756   

### Paper for May 7, 2024:   
iTransformer: Inverted Transformers Are Effective for Time Series Forecasting   
https://arxiv.org/pdf/2310.06625.pdf   

### Paper for April 30, 2024:   
Chronos: Learning the Language of Time Series   
https://arxiv.org/abs/2403.07815  
There is a YouTube on the paper:  
https://www.youtube.com/watch?v=yKKWCqABspw  

### Paper for April 23, 2024:  
Leave No Context Behind: Efficient Infinite Context Transformers with Infini-attention  
https://arxiv.org/pdf/2404.07143.pdf  

### April 16, 2024
From DeepMind, on their generalized AI that can play arbitrary video games,
Scalable Instructable Multiworld Agent (SIMA AI).  
Paper: Scaling Instructable Agents Across Many Simulated Worlds  
https://storage.googleapis.com/deepmind-media/DeepMind.com/Blog/sima-generalist-ai-agent-for-3d-virtual-environments/Scaling%20Instructable%20Agents%20Across%20Many%20Simulated%20Worlds.pdf  
    
Additional resources:  
Deepmind blog:  
https://deepmind.google/discover/blog/sima-generalist-ai-agent-for-3d-virtual-environments/  
2 minute paper:  
https://www.youtube.com/watch?v=5U_Q2Lmnq_c  
Longer YouTube:  
https://www.youtube.com/watch?v=ymKkfRu6dz4  

### April 9, 2024
Cancelled

### April 2, 2024
Evolutionary Optimization of Model Merging Recipes  
https://arxiv.org/pdf/2403.13187.pdf  

### March 26, 2024
Solving Olympiad Geometry Without Human Demonstrations  
https://www.nature.com/articles/s41586-023-06747-5

There are a number of YouTubes, including:  
https://www.youtube.com/watch?v=ZobxevIJQ7A  
and (Yannic)  
https://www.youtube.com/watch?v=ZNK4nfgNQpM

### Tuesday, March 19, 2024
Interpretability in the Wild: a Circuit for Indirect Object Identification in GPT-2 small  
https://arxiv.org/abs/2211.00593  

Additional material - YouTube with the authors (in 2 parts):  
https://www.youtube.com/watch?v=gzwj0jWbvbo  
and  
https://www.youtube.com/watch?v=b9xfYBKIaX4  

Still want more? Two more YouTubes by the YouTube channel owner, Neel Nanda, on his research, inspired by this week's paper.  
(Neel is head of DeepMind's interoperability team - see https://www.neelnanda.io/about)  

https://www.youtube.com/watch?v=m8tzXelUTLo  
and  
https://www.youtube.com/watch?v=tiHRceW-19U  


### Tuesday March 12, 2024
A Review of Sparse Expert Models in Deep Learning  
https://arxiv.org/pdf/2209.01667.pdf  

Background paper:  
Twenty Years of Mixture of Experts  
https://www.ee.hacettepe.edu.tr/~eyuksel/Publications/2012_TwentyYearsofMixtureofExperts.pdf  

### Tuesday, March 5, 2024  
Look Before You Leap: A Universal Emergent Decomposition of Retrieval Tasks in Language Models   
https://arxiv.org/abs/2312.10091  

### Tuesday, February 27, 2024  

Representation Engineering draws on insights from cognitive neuroscience to engineer neural representations, rather than neurons or circuits. Rep. Eng. can be used to apply a control vector during inference to change or limit a model's behavior.  
Paper:  
Representation Engineering - a Top-Down Approach to AI Transparency  
https://arxiv.org/pdf/2310.01405.pdf   

Additional background info:  
Blog:  
Representation Engineering Mistral-7B an Acid Trip  
https://vgel.me/posts/representation-engineering/  
Another blog:  
Steering GPT-2-XL by adding an activation vector  
https://www.lesswrong.com/posts/5spBue2z2tw4JuDCx/steering-gpt-2-xl-by-adding-an-activation-vector  
Third blog:  
https://www.astralcodexten.com/p/the-road-to-honest-ai  
Github:  
https://github.com/andyzoujm/representation-engineering  
Github - Python library  
https://github.com/vgel/repeng/  

### Tuesday February 20, 2024:  
Grandmaster-Level Chess Without Search  
https://arxiv.org/abs/2402.04494  

### Tuesday, February 13, 2024:  
Mistral 7B  
https://arxiv.org/pdf/2310.06825.pdf  
Mixtral of Experts  
https://arxiv.org/pdf/2401.04088.pdf  

Optional:  
There are many YouTubes on each, including by Yannic Kilcher.  
There are download-and-run llamafile quantized versions of Mistral 7B and Mixtral 8x7B at:  
https://github.com/Mozilla-Ocho/llamafile  
(Mac and Linux, Windows has a few very minor additional steps.)  

### Tueday, February 6, 2024  
Why think step by step - Reasoning emerges from the locality of experience   
https://arxiv.org/pdf/2304.03843.pdf  

### Tuesday, January 30, 2024  
Direct Preference Optimization: Your Language Model is Secretly a Reward Model  
https://arxiv.org/pdf/2305.18290.pdf  

Optional:  
There are lots of YouTubes on DPO to choose from.  

A related github by lucidrains:  
https://github.com/lucidrains/self-rewarding-lm-pytorch  

A couple of more recent related papers:  
Self-Play Fine-Tuning Converts Weak Language Models to Strong Language Models  
https://arxiv.org/pdf/2401.01335v1.pdf  
and  
Self-Rewarding Language Models  
https://arxiv.org/pdf/2401.10020.pdf  

### Tuesday, January 23, 2024  
"Latent Consistency Models: Synthesizing High-Resolution Images with Few-Step Inference  
https://arxiv.org/pdf/2310.04378.pdf     
Additional background items:     
There is at least one YouTube on this paper:  
https://www.youtube.com/watch?v=OT3JWNz0Il8     
Huggingface demos:  
https://huggingface.co/collections/latent-consistency/latent-consistency-model-demos-654e90c52adb0688a0acbe6f     
LCM-LoRA: A Universal Stable-Diffusion Acceleration Module  
https://arxiv.org/abs/2311.05556     

### Tuesday, January 16, 2024  
Consistency Models https://arxiv.org/abs/2303.01469       
There are also multiple YouTubes on Consistency Models.     

### Tuesday, January 9, 2024  
Mamba: Linear-Time Sequence Modeling with Selective State Spaces  
https://arxiv.org/ftp/arxiv/papers/2312/2312.00752.pdf     
A few optional videos (likely are others too):     
Video: https://youtu.be/ouF-H35atOY?si=BFQ_PTVfhfNXBLPb     
Video: https://www.youtube.com/watch?v=ouF-H35atOY     

### Tuesday, January 2, 2024  
The Clock and the Pizza: Two Stories in Mechanistic Explanation of Neural Networks  
https://arxiv.org/pdf/2306.17844.pdf  

___________________________________________________________________________________________________________
# ======== 2023 ========

### Tuesday, November 21, 2023  
paper:  MemGPT -Towards LLMs as an Operating System https://arxiv.org/pdf/2310.08560.pdf     
Blog w MemBPT - https://memgpt.ai/     
youtube:  https://www.youtube.com/watch?v=nQmZmFERmrg     

### Tuesday, November 14, 2023  
paper:   
CLUSTERFORMER: Clustering As A Universal Visual Learner  
https://openreview.net/pdf?id=S1KGaTSOTS 

### Tuesday, November 7, 2023  
paper:  
An Emulator for Fine-Tuning Large Language Models using Small Language Models  
https://arxiv.org/pdf/2310.12962.pdf  

### Tuesday, October 31,  2023   
paper:  
From attribution maps to human-understandable explanations through Concept Relevance Propagation   
https://www.nature.com/articles/s42256-023-00711-8  

### Tuesday, October 24,  2023   
paper:
Liquid Structural State-Space Models   
https://arxiv.org/pdf/2209.12951.pdf   

### Tuesday, October 17, 2023  
paper: 
Liquid Time-Constant Networks  
https://arxiv.org/abs/2006.04439  
youtube:  
https://www.youtube.com/watch?v=IlliqYiRhMU  
shorter video:  
https://www.youtube.com/watch?v=RI35E5ewBuI  

### Tuesday, October 10, 2023   
paper  
3D Gaussian Splatting for Real-Time Radiance Field Rendering  
https://arxiv.org/abs/2308.04079   
youtubes: 
Superb 2 minute video on paper  
https://www.youtube.com/watch?v=HVv_IQKlafQ   
Siggraph 2023 talk on paper - this is 5 minutes  
https://www.youtube.com/watch?v=T_kXY43VZnk&t=3s   
Author's blog, including links to code:  
https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/   

### Tuesday, October 3 , 2023  
paper: https://arxiv.org/abs/2112.04035 
Relating transformers to models and neural representations of the hippocampal formation  
another paper:  
https://amygdala.psychdept.arizona.edu/labspace/JclubLabMeetings/JeanMarc-Build-cognitive-maps.pdf - 
How to build a cognitive map   
YouTubes:  
How Your Brain Organizes Information   
https://www.youtube.com/watch?v=9qOaII_PzGY&t=413s  
Can We Build an Artificial Hippocampus?   
https://www.youtube.com/watch?v=cufOEzoVMVA   
The Tolman-Eichenbaum Machine: Unifying Space and Relational Memory through Generalization in the Hippocampal Formation   
https://www.cell.com/cell/fulltext/S0092-8674(20)31388-X   

### Tuesday, September 26, 2023  
paper:  
3D Gaussian Splatting for Real-Time Radiance Field Rendering   
https://research.nvidia.com/labs/par/Perfusion/   

### Tuesday, September 19, 2023  
paper:  
Imagic: Text-Based Real Image Editing with Diffusion Models   
https://arxiv.org/pdf/2210.09276.pdf   
YouTube:  
https://www.youtube.com/watch?v=PzHMjCtuPuo   
blog:  
https://imagic-editing.github.io/   

### Tuesday, Sept 12, 2023  
LongNet: Scaling Transformers to 1,000,000,000 Tokens   
paper: https://arxiv.org/abs/2307.02486   
Blog:  
https://syncedreview.com/2023/07/10/microsofts-longnet-scales-transformer-to-one-billion-tokens   

### Tuesday, Sept 5, 2023  
Consciousness in Artificial Intelligence: Insights from the Science of Consciousness   
https://arxiv.org/pdf/2308.08708.pdf  

### Tuesday, August 29, 2023  
paper:  
A Theory for Emergence of Complex Skills in Language Models and video   
https://arxiv.org/pdf/2307.15936.pdf  
youtube:  
https://www.youtube.com/watch?v=0D23NeBjCeQ   

### Tuesday, August 22, 2023  
Paper: Neural Laplace: Learning diverse classes of differential equations in the Laplace domain   
https://arxiv.org/pdf/2206.04843.pdf
Slides and video from ICML 2022:  
https://icml.cc/virtual/2022/oral/16728   

### Wednesday, August 16, 2023  
paper: https://arxiv.org/abs/2308.03296 - Studying Large Language Model Generalization with Influence Functions   
blog: https://www.anthropic.com/index/influence-functions   

### Wednesday, August 9, 2023  
paper: Music Generations https://arxiv.org/pdf/2306.05284.pdf   
blog: https://about.fb.com/news/2023/08/audiocraft-generative-ai-for-music-and-audio/   
blog: https://ai.meta.com/blog/audiocraft-musicgen-audiogen-encodec-generative-ai-audio/  

### Wednesday, August 2, 2023   
paper: https://arxiv.org/abs/2205.10343 Towards Understanding Grokking: An Effective Theory of Representation Learning   
blog: https://ericjmichaud.com/grokking-squared/  
blog: https://www.beren.io/2022-01-11-Grokking-Grokking/   
blog: https://www.beren.io/2022-04-17-Understanding_Overparametrized_Generalization/   

### Wednesday, July 26, 2023  
paper: Mixture of experts (similar to chatGPT4): https://arxiv.org/abs/2305.14705  

blog: Mixture-of-Experts with Expert Choice Routing -   
https://ai.googleblog.com/2022/11/mixture-of-experts-with-expert-choice.html  

blot: Introducing Pathways: A next-generation AI architecture  
https://blog.google/technology/ai/introducing-pathways-next-generation-ai-architecture/   

### Wednesday, July 19, 2023  
We're going to cover Chapter 16 Deep Networks for Classification  from the following book:  
https://book-wright-ma.github.io/Book-WM-20210422.pdf - High dimensional Data Analysis with Low Dimensional Models
blog:  https://terrytao.wordpress.com/2007/04/13/compressed-sensing-and-single-pixel-cameras/#more-25  

### Wednesday, July 12, 2023  
We're going to cover the 4th chapter of this book.   
https://book-wright-ma.github.io/Book-WM-20210422.pdf - High dimensional Data Analysis with Low Dimensional Models  

### Wednesday, July 5, 2023  
We're going to cover the 1st chapter of this book.  
https://book-wright-ma.github.io/Book-WM-20210422.pdf - High dimensional Data Analysis with Low Dimensional Models  
Blog:  https://terrytao.wordpress.com/2007/04/13/compressed-sensing-and-single-pixel-cameras/#more-25  

### Wednesday, June 28, 2023  
paper: https://arxiv.org/pdf/2305.17126.pdf - Large Language Models as Tool Makers  
youtube:  https://www.youtube.com/watch?v=qWI1AJ2nSDY   
youtube:  https://www.youtube.com/watch?v=KXlPzMRTfMk   
youtube:  https://www.youtube.com/watch?v=srDVNbxPgZI   

### Wednesday, June 21, 2023  
Consciousness as a Memory System https://pubmed.ncbi.nlm.nih.gov/36178498/  

### Wednesday, June 14, 2023  
https://arxiv.org/abs/1804.08838   
Blog: https://www.uber.com/blog/intrinsic-dimension/   
more good stuff on intrinsic dimension:   
Nature paper: https://www.nature.com/articles/s41598-017-11873-y   
Wikipedia: https://en.wikipedia.org/wiki/Intrinsic_dimension   
Application - Yann LeCun at 57:15 on does text fully represent world model?   
https://www.youtube.com/watch?v=SGzMElJ11Cc   
vs. differing view from Ilya Sutskever at 15:30   
https://www.youtube.com/watch?v=SjhIlw3Iffs    
Applying intrinsic dimension to scaling laws in training / loss:   
https://jmlr.csail.mit.edu/papers/volume23/20-1111/20-1111.pdf   
https://arxiv.org/abs/2102.06701   

### Wednesday, June 7, 2023  
Paper:  https://arxiv.org/pdf/2305.16291.pdf   
Twit:  Tweet with nice overview by author https://twitter.com/DrJimFan/status/1662117784023883777   
Code:   https://github.com/MineDojo/Voyager   
website:  https://voyager.minedojo.org/ 

### Wednesday, May 31, 2023   
paper:  https://arxiv.org/pdf/2203.15556.pdf - Training Compute-Optimal Large Language Models  
blog:  https://www.lesswrong.com/posts/6Fpvch8RR29qLEWNH/chinchilla-s-wild-implications   
blog:  https://www.harmdevries.com/post/model-size-vs-compute-overhead/   
google blog:  https://www.cnbc.com/2023/05/16/googles-palm-2-uses-nearly-five-times-more-text-data-than-predecessor.html  

### Wednesday, May 24, 2023  
paper:  https://arxiv.org/abs/2212.09720 - The case for 4-bit precision: k-bit Inference Scaling Laws  
paper:  https://arxiv.org/pdf/2210.17323.pdf - GPTQ: ACCURATE POST-TRAINING QUANTIZATION FOR GENERATIVE PRE-TRAINED TRANSFORMERS  

### Wednesday, May 17, 2023
paper: https://arxiv.org/pdf/2106.09685.pdf - LORA: LOW-RANK ADAPTATION OF LARGE LANGUAGE MODELS  

### Wednesday, May 10, 2023   
paper: https://arxiv.org/pdf/2210.03629.pdf - REACT: SYNERGIZING REASONING AND ACTING IN LANGUAGE MODELS   
paper: https://www.pinecone.io/learn/locality-sensitive-hashing/   

### Wednesday, May 3, 2023   
paper: https://arxiv.org/pdf/2201.11903.pdf - Chain of thought prompting elicits reasoning in large language models.   
paper: https://arxiv.org/pdf/2210.03629.pdf - REACT: SYNERGIZING REASONING AND ACTING IN LANGUAGE MODELS   
paper: https://www.pinecone.io/learn/locality-sensitive-hashing/   

### Wednesday, Apr 26, 2023  
https://python.langchain.com/en/latest/modules/agents.html  
https://arxiv.org/pdf/2210.03629.pdf - REACT: SYNERGIZING REASONING AND ACTING IN LANGUAGE MODELS  
https://www.pinecone.io/learn/locality-sensitive-hashing/  

### Wednesday, Apr 19, 2023 
Blog:  https://yoheinakajima.com/task-driven-autonomous-agent-utilizing-gpt-4-pinecone-and-langchain-for-diverse-applications/   
Code:  https://github.com/hwchase17/langchain   

### Wednesday, Apr 12, 2023  
Paper:  Eliciting Latent Predictions from Transformers with the Tuned Lens https://arxiv.org/abs/2303.08112 

### Wednesday, Apr 5, 2023  
Paper:  https://openreview.net/pdf?id=lMMaNf6oxKM - Recipe for a General, Powerful, Scalable Graph Transformer  
youtube: https://www.youtube.com/watch?v=DiLSCReBaTg  

### Wednesday, Mar 29, 2023  
Paper: https://proceedings.neurips.cc/paper/2021/hash/f1c1592588411002af340cbaedd6fc33-Abstract.html - Do Transformers Really Perform Badly for Graph Representation?   
video: https://www.youtube.com/watch?v=FKuQpPIRjLk - review by authors  
video: https://www.youtube.com/watch?v=xQ5ltOOxoFg   

### Wednesday, Mar 22, 2023
Paper: https://arxiv.org/abs/2212.07359 - Post-hoc Uncertainty Learning using a Dirichlet Meta-Model  
youtube: https://www.youtube.com/watch?v=nE8XJ1f0zO0  

### Wednesday, Mar 15, 2023  
Paper: https://arxiv.org/abs/2202.05262 - Locating and Editing Factual Associations in GPT  
blog:  https://rome.baulab.info/  
Yannic video: https://www.youtube.com/watch?v=_NMQyOu2HTo  

### Wednesday, Mar 8, 2023  
Paper: Human-Timescale Adaptation in an Open-Ended Task Space: https://arxiv.org/pdf/2301.07608.pdf  
https://www.youtube.com/watch?v=A2hOWShiYoM   
https://sites.google.com/view/adaptive-agent/  

### Wednesday, Mar 1, 2023  
Paper: Toolformer: Language Models Can Teach Themselves to Use Tools: https://arxiv.org/abs/2302.04761   

### Wednesday, Feb 22, 2023  
Paper: https://arxiv.org/pdf/2203.02155.pdf - Training language models to follow instructions with human feedback  

### Wednesday, Feb 15, 2023  
Paper:  https://arxiv.org/pdf/2111.15664.pdf - OCR-free Document Understanding Transformer  

### Wednesday, Feb 8, 2023  
Paper: https://arxiv.org/abs/2205.06175 - A generalist agent - Gato  
YouTube: Eden Mayer https://www.youtube.com/watch?v=wSQJZHfAg18   
YouTube - Jay Alamar https://www.youtube.com/watch?v=kT6DYKgWNHg  
YouTube - Lex Fridman and Oriol Vinyals on How Gato Works https://www.youtube.com/watch?v=vwB9zO2h9j0  
Overview - main site on Gato at Deepmind https://www.deepmind.com/publications/a-generalist-agent  
blog review - https://arshren.medium.com/deep-minds-generalist-agent-gato-209969e12782   

### Wednesday, Feb 1, 2023  
Paper:  https://openreview.net/pdf?id=M95oDwJXayG - ADDRESSING PARAMETER CHOICE ISSUES IN UNSUPERVISED DOMAIN ADAPTATION BY AGGREGATION  

### Wednesday, Jan 25, 2023  
Paper: https://arxiv.org/pdf/2301.04104v1.pdf - Mastering Diverse Domains through World Models  
Blog:  https://danijar.com/project/dreamerv3/  
YouTube:  https://www.youtube.com/watch?v=vfpZu0R1s1Y   

### Wednesday, Jan 18, 2023  
Paper:  https://arxiv.org/abs/2212.04089 - Composable NN: Editing Models With Task Arithmetic  

### Wednesday, Jan 11, 2023  
Paper:  https://arxiv.org/pdf/1707.06690.pdf - DeepPath: A Reinforcement Learning Method for Knowledge Graph Reasoning  

### Wednesday, Jan 4, 2023   
Paper: https://arxiv.org/abs/2212.04458 - GENERAL-PURPOSE IN-CONTEXT LEARNING BY META-LEARNING TRANSFORMERS  

___________________________________________________________________________________________________________
# ======== 2022 ========

### Wednesday, Dec 21, 2022  
paper:  https://arxiv.org/pdf/2209.04836.pdf - Git Re-Basin: Merging Models modulo Permutation Symmetries  

### Wednesday, Dec 14, 2022  
paper: https://arxiv.org/abs/2012.09855 - Infinite Nature: Perpetual View Generation of Natural Scenes from a Single Image  
blog: https://infinite-nature.github.io/  

### Wednesday, Dec 7, 2022  
Paper: https://arxiv.org/abs/2206.00364 - Elucidating the Design Space of Diffusion-Based Generative Models  
video: https://www.youtube.com/watch?v=OYiQctx7kDE   

### Wednesday, Nov 30, 2022  
paper: https://arxiv.org/pdf/2206.10991.pdf - Graph Neural Networks as Gradient Flows: understanding graph convolutions via energy  
youtube (author):  https://www.youtube.com/watch?v=sgTTtmwOMgE  
youtube:   https://www.youtube.com/watch?v=hmI4C6AodEQ   

### Wednesday, Nov 16, 2022  
paper: https://www.pnas.org/doi/full/10.1073/pnas.2016239118   
video: https://slideslive.com/38942412/biological-structure-and-function-emerge-from-scaling-unsupervised-learning-to-250-million-protein-sequences  

### Wednesday, Nov 9, 2022  
paper: https://arxiv.org/pdf/2209.11178.pdf - Poisson Flow Generative Models   

### Wednesday, Nov 2, 2022  
paper:  https://arxiv.org/pdf/2209.12892.pdf - LEARNING TO LEARN WITH GENERATIVE MODELS OF NEURAL NETWORK CHECKPOINTS  
blog: https://www.marktechpost.com/2022/10/21/latest-machine-learning-research-at-uc-berkeley-proposes-a-way-to-design-a-learned-optimizer-using-generative-models-of-neural-network-checkpoints/   
author blog:  https://www.wpeebles.com/Gpt.html  

### Wednesday, Oct 26, 2022  
paper:  Cellular automata as convolutional neural networks https://arxiv.org/pdf/1809.02942.pdf  
survey: Collective Intelligence for Deep Learning: A Survey of Recent Developments https://arxiv.org/abs/2111.14377  
demo:  Self-classifying MNIST Digits https://distill.pub/2020/selforg/mnist/  

### Wednesday, Oct 19, 2022   
paper:  https://proceedings.mlr.press/v162/zhu22c/zhu22c.pdf - Neural-Symbolic Models for Logical Queries on Knowledge Graphs  

### Wednesday, Oct 12, 2022  
paper:  https://arxiv.org/pdf/2206.02768.pdf - The Neural Covariance SDE: Shaped Infinite Depth-and-Width Networks at Initialization  

### Wednesday, Oct 5, 2022  
paper: https://papers.nips.cc/paper/2019/file/952285b9b7e7a1be5aa7849f32ffff05-Paper.pdf - Legendre Memory Units: Continuous-Time   

### Wednesday, Sept 28, 2022  
paper: https://arxiv.org/pdf/2208.01618.pdf - An Image is Worth One Word: Personalizing Text-to-Image Generation using Textual Inversion   
githup.io: https://textual-inversion.github.io/    
YouTube https://www.youtube.com/watch?v=f3oXa7_SYek   

### Wednesday, Sept 21, 2022  
paper:  https://arxiv.org/pdf/2205.14415.pdf - Non-stationary Transformers: Rethinking the Stationarity in Time Series Forecasting  

### Wednesday, Sept 14, 2022  
paper: https://arxiv.org/abs/2110.02402 - Language Modeling using LMUs: 10x Better Data Efficiency or Improved Scaling Compared to Transformers   
youtube vid: https://www.youtube.com/watch?v=8t64QaTdBcU  

### Wednesday, August 31, 2022  
Paper: HOW NEURAL NETWORKS EXTRAPOLATE: FROM FEEDFORWARD TO GRAPH NEURAL NETWORKS - https://arxiv.org/pdf/2009.11848.pdf  

### Wednesday, August 24, 2022  
Paper:  Masked Siamese Networks for Label-Efficient Learning - https://arxiv.org/abs/2204.07141  

### Wednesday, August 17, 2022  
Paper:  Principle of Maximal Coding Rate Reduction https://arxiv.org/abs/2006.08558  
ReduNet:  https://arxiv.org/pdf/2105.10446.pdf   
Github:  https://github.com/ryanchankh/mcr2   

### Wednesday, August 10, 2022  
Paper:  On the Principles of Parsimony and Self-Consistency for the Emergence of Intelligence https://arxiv.org/abs/2207.04630   
Background: On the Principles of Parsimony and Self-Consistency for the Emergence of Intelligence https://arxiv.org/abs/2207.04630   
Background:  https://www.youtube.com/watch?v=OIVcfZeR1CE  youtube by author   
Background:   https://cmsa.fas.harvard.edu/wp-content/uploads/2021/04/Deep_Networks_from_First_Principles.pdf  -  slides by author  


### Wednesday, August 3, 2022   
Paper:  Data Distributional Properties Drive Emergent In-Context Learning in Transformers https://arxiv.org/pdf/2205.05055.pdf 

### Wednesday, July 27, 2022  
Paper: A Mathematical Framework for Transformer Circuits https://transformer-circuits.pub/2021/framework/index.html#model-simplifications  

### Wednesday, July 20, 2022  
Paper: A Mathematical Framework for Transformer Circuits https://transformer-circuits.pub/2021/framework/index.html#model-simplifications  

### Wednesday, July 13, 2022  
Paper: https://arxiv.org/abs/2001.08361 - Scaling Laws for Neural Language Models   
Blog: https://medium.com/nlplanet/two-minutes-nlp-scaling-laws-for-neural-language-models-add6061aece7   

### Wednesday, July 6, 2022  
Paper: https://arxiv.org/abs/2206.11795 - Video PreTraining (VPT): Learning to Act by Watching Unlabeled Online Videos  
https://github.com/openai/Video-Pre-Training  
Yannic Review:  https://www.youtube.com/watch?v=oz5yZc9ULAc  

### Wednesday, June 29, 2022  
Paper:  https://arxiv.org/pdf/2110.00966.pdf - Translating Images into Maps  

### Wednesday, June 22, 2022
Paper: https://arxiv.org/abs/2205.09665 - Automated Crossword Solving

### Wednesday, June 15, 2022   
Paper: https://arxiv.org/pdf/2205.10824.pdf - ReLU Fields: The Little Non-linearity That Could   


### Wednesday, June 8, 2022  
Paper: https://arxiv.org/abs/2102.06810 - Understanding Self-Supervised Learning Dynamics without Contrastive Pairs   

### Wednesday, June 1, 2022   
Paper: https://arxiv.org/pdf/2205.06175.pdf - A Generalist Agent  
Blog: https://www.deepmind.com/publications/a-generalist-agent  

### Wednesday, May 25, 2022   
https://arxiv.org/pdf/2202.05780.pdf - A Modern Self-Referential Weight Matrix That Learns to Modify Itself  

### Wednesday, May 18, 2022  
https://openreview.net/pdf?id=M752z9FKJP - LEARNING STRIDES IN CONVOLUTIONAL NEURAL NETWORKS   

### Wednesday, May 11, 2022  
https://openreview.net/pdf?id=b-ny3x071E5 - BOOTSTRAPPED META-LEARNING   

### Wednesday, May 4, 2022  
https://arxiv.org/abs/2202.06991 - Transformer Memory as a Differentiable Search Index   
https://www.youtube.com/watch?v=C7mUYocWdG0 - Yannic author interview   
https://www.youtube.com/watch?v=qlB0TPBQ7YY - Yannic on Transformer paper   

### Wednesday, April 27, 2022  
https://arxiv.org/abs/2204.06125 - Hierarchical Text-Conditional Image Generation with CLIP Latents  
https://openai.com/dall-e-2/ - OpenAI blog  
https://www.youtube.com/watch?v=j4xgkjWlfL4 - yannic video  

### Wednesday, April 20, 2022 
https://arxiv.org/pdf/2103.00020.pdf - Learning Transferable Visual Models From Natural Language Supervision  
https://www.youtube.com/watch?v=1LUWWAnK_Ks  
https://www.youtube.com/watch?v=3X3EY2Fgp3g  

### Wednesday, April 13, 2022
https://arxiv.org/pdf/2110.13985.pdf - Combining Recurrent, Convolutional, and Continuous-time
Models with Linear State-Space Layers

### Wednesday, April 6, 2022
https://arxiv.org/pdf/2202.00666.pdf - Typical Decoding for Natural Language Generation

https://youtu.be/_EDr3ryrT_Y 

https://www.youtube.com/watch?v=AvHLJqtmQkE 

### Wednesday, March 30, 2022
https://arxiv.org/pdf/2105.04906.pdf - VICREG: VARIANCE-INVARIANCE-COVARIANCE REGULARIZATION FOR SELF-SUPERVISED LEARNING   
https://www.youtube.com/watch?v=MzKDNmOJ67Q  

### Wednesday, March 23, 2022 
https://openreview.net/forum?id=4orlVaC95Bo - Task-Agnostic Undesirable Feature Deactivation Using Out-of-Distribution Data

### Wednesday, March 16, 2022
https://arxiv.org/abs/2203.03466 - Tensor Programs V: Tuning Large Neural Networks via Zero-Shot Hyperparameter Transfer  
https://www.youtube.com/watch?v=MNOJQINH-qw  

### Wednesday, March 9, 2022 
https://arxiv.org/abs/2201.12122 - Can Wikipedia Help Offline Reinforcement Learning?   
Yannic's talk on this,  
https://www.youtube.com/watch?v=XHGh19Hbx48   
and he also has a followon video interview with the authors   
https://www.youtube.com/watch?v=FNDVy_BR8aA   


### Wednesday, March 2, 2022 - 
https://arxiv.org/pdf/2107.03342.pdf - A Survey of Uncertainty in Deep Neural Networks

### Wednesday, February 23, 2022 - 
https://arxiv.org/pdf/2201.08239v2.pdf - LaMDA: Language Models for Dialog Applications

### Wednesday, February 16, 2022 - 
https://openreview.net/pdf?id=TrjbxzRcnf- MEMORIZING TRANSFORMERS

### Wednesday, February 9, 2022 - 
https://arxiv.org/pdf/2106.07644.pdf - A Continuized View on Nesterov Acceleration for Stochastic Gradient Descent and Randomized Gossip

### Wednesday, February 2, 2022 - 
https://arxiv.org/pdf/2108.08052.pdf - Moser Flow: Divergence-based Generative Modeling on Manifolds

### Wednesday, January 26, 2022 - 
https://dylandoblar.github.io/noether-networks/ - Noether Networks: meta-learning useful conserved quantities

https://www.youtube.com/watch?v=Xp3jR-ttMfo

### Wednesday, January 19, 2022 - 
https://arxiv.org/pdf/2010.15277.pdf - Class-incremental learning: survey and performance evaluation on image classification

### Wednesday, January 12, 2022 - 
https://arxiv.org/abs/2006.11287 - Discovering Symbolic Models from Deep Learning with Inductive Biases 

### Wednesday, January 5, 2022 - 
https://arxiv.org/pdf/2006.09252.pdf - Improving Graph Neural Network Expressivity via Subgraph Isomorphism Counting

___________________________________________________________________________________________________________
## ======== 2021 ========

### Wednesday, December 29, 2021 - 
https://arxiv.org/pdf/2112.04426.pdf - Improving Language Models by Retrieving from Trillions of Tokens

https://www.deepmind.com/research/publications/2021/improving-language-models-by-retrieving-from-trillions-of-tokens

### Wednesday, December 22, 2021 - 
https://arxiv.org/abs/2106.01798 - Implicit MLE: Backpropagating Through Discrete Exponential Family Distributions

https://www.youtube.com/watch?v=W2UT8NjUqrk

### Wednesday, December 15, 2021 - 
https://arxiv.org/pdf/2108.01073.pdf - Image Synthesis and Editing with Stochastic Differential Equations

### Wednesday, December 1, 2021 - 
https://openreview.net/forum?id=HfpNVDg3ExA
OpenReviewOpenReview
Probabilistic Transformer For Time Series Analysis

### Wednesday, November 17, 2021 - 
https://arxiv.org/pdf/2110.03922.pdf - NEURAL TANGENT KERNEL EIGENVALUES ACCURATELY PREDICT GENERALIZATION

### Wednesday, November 10, 2021 - 
https://arxiv.org/pdf/2104.00681.pdf - NeuralRecon: Real-Time Coherent 3D Reconstruction from Monocular Video

https://github.com/zju3dv/NeuralRecon


### Wednesday, October 27, 2021 - 
https://arxiv.org/pdf/2110.09485.pdf - Learning in High Dimension Always Amounts to Extrapolation

### Wednesday, October 20, 2021 - 
https://arxiv.org/pdf/2109.02355.pdf - A Farewell to the Bias-Variance Tradeoff? An Overview of the Theory of Overparameterized Machine Learning

### Wednesday, October 13, 2021 - 
https://arxiv.org/pdf/2006.09011.pdf - Improved Techniques for Training Score-Based Generative Models

### Wednesday, October 6, 2021 - 
https://arxiv.org/abs/2006.05929 - Dataset Condensation with Gradient Matching

### Wednesday, September 29, 2021 - 
https://arxiv.org/abs/1811.10959 - Dataset distillation

### Wednesday, September 22, 2021 - 
https://arxiv.org/pdf/2003.13216.pdf - Learning to Learn Single Domain Generalization

### Wednesday, September 15, 2021 - 
https://arxiv.org/pdf/2108.11482.pdf - ETA Prediction with Graph Neural Networks in Google Maps

### Wednesday, September 8, 2021 - 
https://cascaded-diffusion.github.io/assets/cascaded_diffusion.pdf - Cascaded Diffusion Models for High Fidelity Image Generation

### Wednesday, September 1, 2021 - 
https://arxiv.org/pdf/2107.06277.pdf - Why Generalization in RL is Difficult: Epistemic POMDPs and Implicit Partial Observability

### Wednesday, August 25, 2021 - 
https://arxiv.org/abs/2108.07732 - Program Synthesis with Large Models

### Wednesday, August 18, 2021 - 
https://arxiv.org/abs/2012.13349 - Solving Mixed Integer Programs Using Neural Networks

### Wednesday, August 11, 2021 - 
https://www.nature.com/articles/s41586-021-03819-2 - DeepFold

### Wednesday, August 4, 2021 - 
Alphafold - blog https://deepmind.com/blog/article/alphafold-a-solution-to-a-50-year-old-grand-challenge-in-biology paper https://www.nature.com/articles/s41586-021-03819-2 supplemental info https://static-content.springer.com/esm/art%3A10.1038%2Fs41586-021-03819-2/MediaObjects/41586_2021_3819_MOESM1_ESM.pdf

### Wednesday, July 21, 2021 - 
https://www.zdnet.com/article/googles-supermodel-deepmind-perceiver-is-a-step-on-the-road-to-an-ai-machine-that-could-process-everything/ https://arxiv.org/abs/2103.03206

### Wednesday, July 14, 2021 - 
https://arxiv.org/pdf/1503.03585.pdf (Deep Unsupervised Learning using Non equilibrium Thermodynamics) by Surya Ganguli at Stanford
##  
Wednesday, July 7, 2021 - 
https://arxiv.org/pdf/2105.05233.pdf - Diffusion Models Beat GANs on Image Synthesis

### Wednesday, June 30, 2021 - 
https://arxiv.org/pdf/2006.11239.pdf - Denoising Diffusion Probabilistic Models

### Wednesday, June 23, 2021 - 
https://arxiv.org/abs/2010.03409 - Learning mesh-based simulation with graph networks

https://sites.google.com/view/learning-to-simulate

https://deepmind.com/research/publications/Learning-to-Simulate-Complex-Physics-with-Graph-Networks

### Wednesday, June 16, 2021 - 
https://arxiv.org/pdf/2106.01345.pdf - Decision Transformer: Reinforcement Learning via Sequence Modeling

https://www.youtube.com/watch?v=-buULmf7dec

https://sites.google.com/berkeley.edu/decision-transformer

### Wednesday, June 9, 2021 - 
https://arxiv.org/pdf/2103.07945.pdf - Learning One Representation to Optimize All Rewards

### Wednesday, June 2, 2021 - 
https://distill.pub/2021/multimodal-neurons/ - Multimodal Neurons in Artificial Neural Networks

https://openai.com/blog/clip/ - CLIP: Connecting Text and Images

### Wednesday, May 26, 2021 - 
https://arxiv.org/pdf/2104.14294.pdf - Emerging Properties in Self-Supervised Vision Transformers

https://ai.facebook.com/blog/dino-paws-computer-vision-with-self-supervised-transformers-and-10x-more-efficient-training/

### Wednesday, May 19, 2021 - 
https://arxiv.org/pdf/2104.10558.pdf - Contingencies from Observations: Tractable ContingencyPlanning with Learned Behavior Models

### Wednesday, May 12, 2021 - 
https://arxiv.org/pdf/1806.09055.pdf - DARTS: Differentiable Architecture Search (ICLR 2019)

### Wednesday, May 5, 2021 - 
https://arxiv.org/pdf/2104.06644.pdf - Masked Language Modeling and the Distributional Hypothesis:Order Word Matters Pre-training for Little

### Wednesday, April 28, 2021 - 
https://arxiv.org/pdf/2009.03717.pdf - Hierarchical message passing graph neural networks

### Wednesday, April 14, 2021 - 
https://arxiv.org/pdf/2103.03230v1.pdf - Barlow Twins: Self-Supervised Learning via Redundancy Reduction

### Wednesday, April 7, 2021 - 
https://arxiv.org/pdf/2103.14770.pdf - Categorical representation learning: morphism is all you need

### Wednesday, March 31, 2021 - 
https://arxiv.org/pdf/2102.12736v1.pdf - Time-Series Imputation with Wasserstein Interpolation for Optimal Look-Ahead-Bias and Variance Tradeoff

### Wednesday, March 24, 2021 - 
https://awacrl.github.io/ - Accelerating online reinforcement learning with offline datasets

### Wednesday, March 17, 2021 - 
https://arxiv.org/pdf/2102.12092.pdf - Zero-Shot Text-to-Image Generation

https://openai.com/blog/dall-e/

### Wednesday, March 10, 2021 - 
https://giotto-ai.github.io/gtda-docs/latest/notebooks/gravitational_waves_detection.html

### Wednesday, March 3, 2021 - 
https://arxiv.org/pdf/2102.08602.pdf - Modeling long-range interactions without attention

### Wednesday, February 24, 2021 - 
https://arxiv.org/pdf/2101.08692.pdf - Characterizing signal propagation to close the performance gap in unnormalized resnets

### Wednesday, February 17, 2021 - 
https://arxiv.org/pdf/2006.10742.pdf - Learning Invariant Representations forReinforcement Learning without Reconstruction

### Wednesday, February 10, 2021 - 
https://arxiv.org/pdf/2007.13544.pdf - Combining Deep Reinforcement Learning and Search for Imperfect-Information Games

### Wednesday, February 3, 2021 - 
https://arxiv.org/pdf/2010.11929.pdf - An image is worth 16x16 words: transformers for image recognition at scale

### Wednesday, January 27, 2021 - 
https://arxiv.org/abs/2003.02821 - What went wrong and when? Instance-wise feature importance for time-series black-box models

### Wednesday, January 20, 2021 - 
https://arxiv.org/pdf/1912.09363.pdf - Temporal Fusion Transformersfor Interpretable Multi-horizon Time Series Forecasting

### Wednesday, January 13, 2021 - 
https://arxiv.org/abs/1905.10403 - Neural Jump Stochastic Differential Equations

### Wednesday, January 6, 2021 - 
http://implicit-layers-tutorial.org/neural_odes/ - We're continuing this from last week. This week we'll cover Ch 3,4,5.

___________________________________________________________________________________________________________
# ======== 2020 ========

### Wednesday, December 30, 2020 - 
http://implicit-layers-tutorial.org/ - NeurIPS tutorial on deep implicit networks

### Wednesday, December 23, 2020 - 
https://arxiv.org/pdf/1907.03907.pdf - Latent ODEs for Irregularly-Sampled Time Series

https://www.youtube.com/watch?v=tOkH339Wucs

### Wednesday, December 16, 2020 - 
https://papers.nips.cc/paper/2020/file/08425b881bcde94a383cd258cea331be-Paper.pdf - Ridge Rider: Finding Diverse Solutions by FollowingEigenvectors of the Hessian

### Wednesday, December 9, 2020 - 
https://proceedings.neurips.cc/paper/2020/file/28e209b61a52482a0ae1cb9f5959c792-Paper.pdf
“OOD-MAML: Meta-Learning for Few-Shot Out-of-Distribution Detection and Classification"

### Wednesday, December 2, 2020 - 
https://arxiv.org/pdf/2011.02421.pdf - ONE-SHOT CONDITIONAL AUDIO FILTERING OF ARBITRARY SOUNDS

### Wednesday, November 18, 2020 - 
https://arxiv.org/pdf/2010.14498.pdf - Implicit under-parametrization inhibits data efficient deep reinforcement learning

### Wednesday, October 28, 2020
https://arxiv.org/pdf/2010.03759.pdf - Energy-based Out-of-distribution Detection  
  
### Wednesday, October 21, 2020
https://arxiv.org/abs/2005.01643 - offline reinforcement learning - tutorial review and perspectives on open problems  

### Wednesday, October 14, 2020
https://arxiv.org/pdf/2009.12981.pdf - Parametric UMAP: Learning embeddings with deep neural networks for representation and semi-supervised learning  

### Wednesday, OCtober 7, 2020  
https://arxiv.org/pdf/2009.12981.pdf - Parametric UMAP: Learning embeddings with deep neural networks for representation and semi-supervised learning
Some reference material and a cool movie;
https://arxiv.org/abs/1803.05316
category theory book
https://math.mit.edu/~dspivak/teaching/sp18/
the class
https://www.youtube.com/watch?v=nq6iPZVUxZU    
  
### Wednesday, September 23, 2020  
https://arxiv.org/pdf/2008.02217.pdf - Hopfield Networks is All You Need  
  
### Wednesday, September 9, 2020 and Wednesday September 16, 2020
https://arxiv.org/pdf/1912.02762.pdf - Normalizing Flows for Probabilistic Modeling and Inference  
  
### Wednesday, September 2, 2020  
https://arxiv.org/pdf/2007.02168.pdf - Scalable Differentiable Physics for Learning and Control    

### Wednesday, August 19, 2020  
https://arxiv.org/pdf/1903.11239v3 Tossingbot 
  
### Wednesday, July 22, 2020  
https://arxiv.org/pdf/2002.05709.pdf - A Simple Framework for Contrastive Learning of Visual Representations  
  
### Wednesday, April 15, 2020  
https://arxiv.org/pdf/2002.11089.pdf - Rewriting History with Inverse RL: Hindsight Inference for Policy Improvement    
  
### Mar 11, 2020 - Hacker Dojo  
https://arxiv.org/pdf/2002.11089.pdf - Rewriting History with Inverse RL: Hindsight Inference for Policy Improvement  
  
### Mar 4, 2020 - Hacker Dojo  
https://www.osapublishing.org/DirectPDFAccess/C6D6B2C3-953C-4461-695B6E5E2F993943_415059/prj-7-8-823.pdf?da=1&id=415059&seq=0&mobile=no --Nanophotonic media for artificial neural inference  

### Feb 19, 2020 - Hacker Dojo  
https://arxiv.org/pdf/1910.02789.pdf - Language is Power: Representing States Using Natural Language in Reinforcement Learning  

### Feb 12, 2020 - Hacker Dojo  
https://deepmind.com/blog/article/AlphaFold-Using-AI-for-scientific-discovery - Protein folding paper.

### Feb 5, 2020 - Hacker Dojo 
https://arxiv.org/abs/2001.04451 Reformer, the efficient transformer   
https://ai.googleblog.com/2020/01/reformer-efficient-transformer.html   

### Jan 22, 2020 - Hacker Dojo  
https://arxiv.org/pdf/1906.05717.pdf - Unsupervised Monocular Depth and Ego-motion Learning with Structure and Semantics  

### Jan 15, 2020 - Hacker Dojo
https://arxiv.org/pdf/1912.09524.pdf - Evolving ab initio trading strategies in heterogeneous environments  

### Jan 8, 2020 - Hacker Dojo  
https://arxiv.org/pdf/1911.05892.pdf - Reinforcement Learning for Market Making in Multi-agent Dealer Market  

___________________________________________________________________________________________________________
# ======== 2019 ========

### Dec 18, 2019 - Hacker Dojo
https://www.nature.com/articles/s41586-019-1724-z.epdf?author_access_token=lZH3nqPYtWJXfDA10W0CNNRgN0jAjWel9jnR3ZoTv0PSZcPzJFGNAZhOlk4deBCKzKm70KfinloafEF1bCCXL6IIHHgKaDkaTkBcTEv7aT-wqDoG1VeO9-wO3GEoAMF9bAOt7mJ0RWQnRVMbyfgH9A%3D%3D   
https://www.gwern.net/docs/rl/2019-vinyals.pdf  
https://deepmind.com/blog/article/AlphaStar-Grandmaster-level-in-StarCraft-II-using-multi-agent-reinforcement-learning  

### Nov 20, 2019 - Hacker Dojo 
https://arxiv.org/pdf/1911.04252.pdf - Self-training with Noisy Student improves ImageNet classification  

### Nov 13, 2019 - Hacker Dojo  
https://arxiv.org/pdf/1910.12713.pdf - Few-shot video-video synthesis  

### Nov 6, 2019 - Hacker Dojo
https://arxiv.org/pdf/1906.11883.pdf - Unsupervised learning of Object Keypoints for Perception and Control  

### Oct 30, 2019 - Hacker Dojo  
https://arxiv.org/pdf/1710.03748.pdf - Emergent Complexity via Multi-Agent Competition  
https://openai.com/blog/competitive-self-play/  

### Oct 23, 2019 - Hacker Dojo  
https://arxiv.org/pdf/1703.04908.pdf - Emergence of Grounded Compositional Language in Multi-Agent Populations  

### Oct 16, 2019 - Hacker Dojo  
https://arxiv.org/pdf/1909.07528.pdf - Emergent tool use from multi agent autocurricula  
https://openai.com/blog/emergent-tool-use/  

### Oct 9, 2019 - Hacker Dojo  
https://arxiv.org/pdf/1901.00949.pdf - Machine Teaching in Hierarchical Genetic Reinforcement Learning: Curriculum Design of Reward Functions for Swarm Shepherding  

### Sept 25, 2019 - Hacker Dojo  
https://arxiv.org/pdf/1812.01729.pdf - Boltzman Generators - Sampling equilibrium states of many body systems with deep learning


### Sept 18, 2019 - Hacker Dojo
https://arxiv.org/pdf/1907.10599.pdf - Fine Grained Spectral Perspective on Neural Networks  

### Sept 11, 2019 - Hacker Dojo  
https://arxiv.org/pdf/1906.08237.pdf - XLNet Generalized autoregressive pretraining for language understanding 

### Sept 4, 2019 - Hacker Dojo  
https://arxiv.org/pdf/1905.09272.pdf - Data efficient image recognition with contrastive predictive coding.  

### August 21, 2019 - Hacker Dojo  
https://arxiv.org/pdf/1904.10509.pdf - Generating long sequences with sparse transformers 

### August 14, 2019 - Hacker Dojo
https://arxiv.org/pdf/1807.03748.pdf - Representation learning with contrastive predictive coding.  

### July 31, 2019 - Hacker Dojo
https://arxiv.org/pdf/1906.08253.pdf - When to trust your model: model-based policy optimization  

### July 24, 2019 - Hacker Dojo 
https://arxiv.org/pdf/1901.09321.pdf - Fixup initialization - residual learning without normalization  


### July 17, 2019 - Hacker Dojo
http://proceedings.mlr.press/v97/mahoney19a/mahoney19a.pdf  - Traditional and heavy tailed self regularization in neural net models 

### July 3, 2019 - Hacker Dojo 
https://arxiv.org/pdf/1804.08838.pdf - Measuring intrinsic dimension of objective landscapes 

### June 19, 2019 - Hacker Dojo  
https://arxiv.org/abs/1810.09536 - Ordered Neurons: Integrating Tree Structures into Recurrent Neural Networks  

### June 12, 2019 - Hacker Dojo
https://arxiv.org/pdf/1812.05159.pdf - An empirical study of example forgetting during neural network training.  

### June 5, 2019 - Hacker Dojo 
https://arxiv.org/pdf/1812.00417.pdf - Snorkel Drybell - A case study in weak supervision at industrial scale  
https://arxiv.org/pdf/1905.04981.pdf - Modelling instance level annotator reliability for natural language labelling 

### May 29, 2019 - Hacker Dojo  
https://arxiv.org/pdf/1901.09321.pdf - Fixup Initialization: Residual Learning without Normalization  

### May 22, 2019 - Hacker Dojo  
https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf - Language Models are Unsupervised Multitask Learners.  

### May 15, 2019 - Hacker Dojo 
https://arxiv.org/pdf/1811.00995.pdf - Invertible Residual Networks  

### Apr 29, 2019 - Hacker Dojo  
https://arxiv.org/pdf/1904.01681.pdf - Augmented Neural ODE's  

### Apr 8, 2019 - Hacker Dojo  
https://arxiv.org/pdf/1901.00596.pdf - Comprehensive Survey of Graph Neural Nets  
https://github.com/rusty1s/pytorch_geometric  

### Apr 1, 2019 - Hacker Dojo  
https://arxiv.org/pdf/1901.00596.pdf - Comprehensive Survey of Graph Neural Nets  

### Mar 25, 2019 - Hacker Dojo  
https://papers.nips.cc/paper/7539-optimal-algorithms-for-non-smooth-distributed-optimization-in-networks.pdf  - nips award winner

### Mar 18, 2019 - Hacker Dojo 
https://papers.nips.cc/paper/8200-non-delusional-q-learning-and-value-iteration.pdf - Non-delusional Q-learning and Value Iteration  

### Mar 11, 2019 - Hacker Dojo
https://arxiv.org/pdf/1706.03762.pdf - attention is all you need - Vaswani
https://github.com/jadore801120/attention-is-all-you-need-pytorch - easier to read code 
https://www.youtube.com/watch?v=S0KakHcj_rs  
https://tdls.a-i.science/events/2018-10-22/  
https://tdls.a-i.science/events/2019-02-04/  
http://nlp.seas.harvard.edu/2018/04/03/attention.html  


### Mar 4, 2019 - Hacker Dojo 
https://arxiv.org/pdf/1806.02643.pdf - Re-evalating Evaluation

### Feb 25, 2019 - Hacker Dojo
https://arxiv.org/pdf/1812.11951.pdf - Learning to Design RNA  

### Feb 11, 2019 - Hacker Dojo - 
https://arxiv.org/pdf/1901.02860.pdf - Transformer XL - Attentive Language Models, Beyond a fixed length context

### Feb 4, 2019 - Hacker Dojo  
https://arxiv.org/pdf/1809.06646.pdf - Model Free Adaptive Optimal Control of Sequential Manufacturing Process Using Reinforcement Learning  

### January 28, 2019 - Hacker Dojo  
https://arxiv.org/pdf/1806.07366.pdf - Neural Ordinary Differential Equations - Top paper NIPS2019 

### January 21, 2019 - Hacker Dojo 
https://arxiv.org/pdf/1606.05312.pdf - Successor Features for Transfer in Reinforcement Learning  
http://proceedings.mlr.press/v37/schaul15.pdf - Universal Value Function Approximators  
http://proceedings.mlr.press/v80/barreto18a/barreto18a.pdf - Transfer in deep reinforcement learning using successor features and generalised policy improvement.  

https://www.youtube.com/watch?v=YDCPHekLUI4&t=1053s - Tom Schaul  
https://www.youtube.com/watch?v=OCHwXxSW70o - Tejas Kulkarni  


### January 14, 2019 - Hacker Dojo
https://arxiv.org/pdf/1812.07626.pdf - Universal Successor Features Approximators  

### January 7, 2019 - Hacker Dojo
https://arxiv.org/pdf/1810.12715.pdf - On the Effectiveness of Interval Bound Propagation for Training Verifiably Robust Models  

___________________________________________________________________________________________________________
# ======== 2018 ========

### December 17, 2018 - Hacker Dojo
https://openreview.net/pdf?id=S1x4ghC9tQ - Temporal Difference Variational Autoencoder


### December 10, 2018 - Hacker Dojo  
https://openreview.net/pdf?id=S1JHhv6TW - Boosting Dilated Convolution with Mixed Tensor Decompositions  

### December 3, 2018 - Hacker Dojo 
https://arxiv.org/pdf/1712.01208.pdf - The case for learned index structures  

### November 26, 2018 - Hacker Dojo 
https://arxiv.org/abs/1809.07402 - Generalization properties of nn - Socher  
https://einstein.ai/research/blog/identifying-generalization-properties-in-neural-networks - blog for above paper   

### November 19, 2018 - Hacker Dojo  
https://arxiv.org/pdf/1802.05983.pdf - Disentangling by Factorising  
https://arxiv.org/pdf/1804.00104.pdf - Learning Disentangled Joint, Discrete and Continuous Representations  
https://arxiv.org/pdf/1807.05520.pdf - Deep Clustering for Unsupervised Learning of Visual Features  
https://github.com/1Konny/FactorVAE  
https://github.com/paruby/FactorVAE    
https://github.com/nicolasigor/FactorVAE  

### November 12, 2018 - Hacker Dojo
https://arxiv.org/pdf/1810.12894.pdf - Exploration by Random Network Distillation - OpenAI  

### November 5, 2018 - Hacker Dojo 
https://arxiv.org/pdf/1810.04805.pdf - Pre-trainged bi directional transformers for language translation  


### October 22, 2018 - Hacker Dojo  
https://arxiv.org/pdf/1801.02613.pdf - Characterizing Adversarial Examples using Local Intrinsic Dimensionality  


### October 15, 2018 - Hacker Dojo  
https://arxiv.org/pdf/1808.06670.pdf - Learning Deep Representations by Mutual Estimation Estimation and Maximization - Hjelm, Bengio  

### October 8, 2018 - Hacker Dojo  
https://arxiv.org/pdf/1802.04364.pdf - Junction Tree Variational Auto-Encoder for Molecular Graph Generation  
http://snap.stanford.edu/proj/embeddings-www/files/nrltutorial-part2-gnns.pdf  

### October 1, 2018 - Hacker Dojo
https://arxiv.org/pdf/1808.06601.pdf - Video to video synthesis 
https://github.com/NVIDIA/vid2vid - code  

### September 24, 2018 - Hacker Dojo 
https://arxiv.org/pdf/1807.03146.pdf - Discovery of 3d keypoints from 2d image  

### September 17, 2018 - Hacker Dojo 
https://arxiv.org/abs/1709.02371 - PWC-Net: CNNs for Optical Flow Using Pyramid, Warping, and Cost Volume," by Deqing Sun et al. (CVPR 2018) 
Phil Ferrier will present the paper and run though his code for us. Phil's code is on his github reop:  
https://github.com/philferriere/tfoptflow  

### September 10, 2018 - Hacker Dojo
https://arxiv.org/pdf/1807.03247.pdf - Intriguing failure (and improvement) to CNN for determining rotations.  

### September 3, 2018 - Hacker Dojo  
https://arxiv.org/pdf/1803.03324.pdf - Learning Deep Generative Models of Graphs  

### August 27, 2018 - Hacker Dojo
https://arxiv.org/abs/1709.10082 - Optimally decentralized multi-robot collision avoidance w reinforcement learning.  

https://github.com/TensorSwarm/TensorSwarm  - Andreas Pasternak code for above  

### August 13, 2018 - Hacker Dojo
https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/learning-dexterity/learning-dexterity-paper.pdf  -Robot doing single hand manipulations.  
https://www.theverge.com/2018/7/30/17621112/openai-robot-dexterity-dactyl-artificial-intelligence  

### July 30, 2018 - Hacker Dojo - 
https://arxiv.org/pdf/1711.03953.pdf - Breaking the softmax bottleneck  
https://arxiv.org/pdf/1805.10829.pdf - SigSoftMax: Reanalyzing the softmax bottleneck  
https://severelytheoretical.wordpress.com/2018/06/08/the-softmax-bottleneck-is-a-special-case-of-a-more-general-phenomenon/  

### July 23, 2018 - Hacker Dojo - 
https://arxiv.org/pdf/1807.01281.pdf - Human level performance in first person multiplayer games with population reinforcement learning.  
https://deepmind.com/blog/capture-the-flag/ 
https://www.youtube.com/watch?v=steioHoiEms  
https://arxiv.org/abs/1711.09846v2  
https://arxiv.org/pdf/1611.05397.pdf  

### July 16, 2018 - Hacker Dojo 
https://arxiv.org/pdf/1803.10122.pdf - schmidhuber paper on RL  

### July 9, 2018 - Hacker Dojo
https://deepmind.com/research/publications/neural-scene-representation-and-rendering/  - Rendering 3d scene  

### July 2, 2018 - Hacker Dojo - 
https://arxiv.org/pdf/1707.06347.pdf - Proximal Optimization Policies  

### June 25, 2018 - Hacker Dojo  
https://openreview.net/pdf?id=BJOFETxR- - Learning to represent programs with graphs  

### June 18, 2018 - Hacker Dojo  
https://openreview.net/pdf?id=BkisuzWRW - Zero Shot Visual Imitation - Reinforcement Learning  


### June 11, 2018 - Hacker Dojo
https://openreview.net/forum?id=HkL7n1-0b - Wasserstein Auto Encoders - one of ICLR top papers.  

### June 4, 2018 - Hacker Dojo
https://openreview.net/pdf?id=Hy7fDog0b - Ambient GAN - Generative Models from Lossy Measurements - ICLR top paper  


### May 21, 2018 - Hacker Dojo
https://arstechnica.com/science/2018/05/ai-trained-to-navigate-develops-brain-like-location-tracking/  - Grid representations in rat brain  
https://deepmind.com/documents/200/Banino_at_al_final.pdf  --  
https://www.nature.com/articles/s41586-018-0102-6  --  



### May 14, 2018 - Hacker Dojo
https://arxiv.org/pdf/1712.06567.pdf - Deep Neuroevolution: Genetic Algorithms are a Competitive Alternative for
Training Deep Neural Networks for Reinforcement Learning  
https://arxiv.org/pdf/1712.06560.pdf - Improving Exploration in Evolution Strategies for Deep Reinforcement
Learning via a Population of Novelty-Seeking Agents  
https://eng.uber.com/deep-neuroevolution/  - Uber engineering blog post  

### May 7, 2018 - Hacker Dojo
https://arxiv.org/pdf/1801.10130.pdf - spherical CNN  

### Apr 30, 2018 - Hacker Dojo
https://arxiv.org/pdf/1710.07313.pdf - Using machine learning to replicate chaotic attractors  
http://www.bmp.ds.mpg.de/tl_files/bmp/preprints/Zimmermann_Parlitz_preprint.pdf - paper to be published in "chaos"  
https://www.quantamagazine.org/machine-learnings-amazing-ability-to-predict-chaos-20180418/ - blog post  


### Apr 23, 2018 - Hacker Dojo  
https://arxiv.org/pdf/1711.10925.pdf - Deep Image Prior  
https://dmitryulyanov.github.io/deep_image_prior - git hub from authors  
https://box.skoltech.ru/index.php/s/ib52BOoV58ztuPM  
http://mlexplained.com/2018/01/18/paper-dissected-deep-image-prior-explained/  
http://fortune.com/2018/04/24/nvidia-artificial-intelligence-images/ - Article w video showing photo editing use  

### Apr 16, 2018 - Hacker Dojo 
Finish Fractal AI  
https://arxiv.org/pdf/1711.07971.pdf - non-local filtering  


### Apr 9, 2018 - Hacker Dojo
http://lanl.arxiv.org/pdf/1803.05049v1 - Fractal AI 

### Apr 2, 2018 - Hacker Dojo 
https://arxiv.org/pdf/1803.04831.pdf - IndRNN longer deeper RNN's  

### Mar 26, 2018 -  Hacker Dojo
https://arxiv.org/pdf/1711.10433.pdf - parallel wavenet  
https://arxiv.org/pdf/1708.04552.pdf - regularizing convnet with cutout (desert paper) 
http://www.cs.toronto.edu/~jmartens/docs/Deep_HessianFree.pdf - will get short presentation on this one.  

### Mar 19, 2018 - Hacker Dojo 
https://arxiv.org/pdf/1802.03268.pdf - Efficient Neural Architecture Search via Parameter Sharing  
https://github.com/carpedm20/ENAS-pytorch 

some related papers and reviews. 
https://arxiv.org/pdf/1708.05344.pdf - One shot architecture search  
https://openreview.net/forum?id=ByQZjx-0-  
and  
https://openreview.net/forum?id=rydeCEhs-  


### Mar 12, 2018 - Hacker Dojo 
https://arxiv.org/abs/1703.10135 - tacotron - end-to-end speech synthesis  
https://arxiv.org/pdf/1712.05884.pdf - tacotron 2  
https://research.googleblog.com/2017/12/tacotron-2-generating-human-like-speech.html - 
https://github.com/A-Jacobson/tacotron2 - pytorch code 
http://research.baidu.com/deep-speech-3%EF%BC%9Aexploring-neural-transducers-end-end-speech-recognition/  

### Feb 26, 2018 - Hacker Dojo  
https://arxiv.org/pdf/1705.09792.pdf - Deep Complex Networks  


### Feb 19, 2018 - Hacker Dojo  
https://arxiv.org/pdf/1801.10308.pdf - Nested LSTM's  
https://arxiv.org/pdf/1705.10142.pdf - KRU from Fair  
https://github.com/hannw/nlstm  - tf code for Nested LSTM

### Feb 12, 2018 - Hacker Dojo  
http://openaccess.thecvf.com/content_cvpr_2017/papers/Khoreva_Simple_Does_It_CVPR_2017_paper.pdf - Weakly Supervised Instance and Semantic Segmentation  
https://www.mpi-inf.mpg.de/departments/computer-vision-and-multimodal-computing/research/weakly-supervised-learning/simple-does-it-weakly-supervised-instance-and-semantic-segmentation/  
https://github.com/philferriere/tfwss - Phil Ferriere's code  
 https://drive.google.com/file/d/1wPHMA4PqygawvIxRiy-2ZMKcpUO447cz/view?usp=sharing - mehul's notebook on segmentation  

### Feb 5, 2018 - Hacker Dojo
https://arxiv.org/pdf/1511.06939.pdf - using rnn for recommendation system  
https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/46488.pdf - latest paper on rnn for recommendation  

### Jan 29, 2018 - Hacker Dojo
https://arxiv.org/pdf/1709.04511.pdf - Empirical study of multi-agent RL  
https://github.com/geek-ai/1m-agents - code 

### Jan 22, 2018 - Hacker Dojo  
https://arxiv.org/pdf/1704.00028.pdf - Improvements in Wasserstein GAN training  

### Jan 15, 2018 - Hacker Dojo

https://arxiv.org/pdf/1710.02298.pdf - Combining improvements in deep reinforcement learning  

### Jan 8, 2018 - Hacker Dojo
https://openreview.net/pdf?id=HJWLfGWRb - follow-on to capsule network paper  
https://www.youtube.com/watch?v=pPN8d0E3900  
https://www.youtube.com/watch?v=2Kawrd5szHE  
https://github.com/ageron/handson-ml/blob/master/extra_capsnets.ipynb  
https://github.com/naturomics/CapsNet-Tensorflow  
https://medium.com/ai%C2%B3-theory-practice-business/understanding-hintons-capsule-networks-part-ii-how-capsules-work-153b6ade9f66  

___________________________________________________________________________________________________________
# ======== 2017 ========

### Dec 11, 2017 - Hacker Dojo  
https://arxiv.org/pdf/1710.09829.pdf - Dynamic routing between capsules - Hinton  

### Nov 27, 2017 - Hacker Dojo
https://arxiv.org/pdf/1701.01724.pdf - DeepStack: Expert-Level Artificial Intelligence in
Heads-Up No-Limit Poker  

### Nov 13, 2017 - Hacker Dojo
https://deepmind.com/documents/119/agz_unformatted_nature.pdf - alpha zero paper  
https://webdocs.cs.ualberta.ca/~mmueller/talks/2016-LeeSedol-AlphaGo.pdf  - some slides  


### Nov 6, 2017 - Hacker Dojo  
https://arxiv.org/pdf/1703.10593.pdf - cycle consistent GANs  

### Oct 30, 2017 - Hacker Dojo  
https://arxiv.org/pdf/1503.02406.pdf  Naftali Tishby and Noga Zaslavsky. information bottleneck principle.  

https://www.cs.huji.ac.il/labs/learning/Papers/allerton.pdf - Naftali Tishby, Fernando C. Pereira, and William Bialek. The information bottleneck method. 

https://www.reddit.com/r/MachineLearning/comments/75uua6/r_2_hr_talk_information_theory_of_deep_learning/  

### Oct 23, 2017 - Hacker Dojo  

Mask R-CNN  
https://arxiv.org/abs/1703.06870  


And these are prerequisites (read at least Fast R-CNN and Faster R-CNN)  

R-CNN  
https://arxiv.org/abs/1311.2524  

Fast R-CNN  
https://arxiv.org/pdf/1504.08083.pdf  

Faster R-CNN  
https://arxiv.org/abs/1506.01497 Feature Pyramid Networks  
https://arxiv.org/abs/1612.03144  


### Oct 16, 2017 - Hacker Dojo 
https://arxiv.org/pdf/1703.00810.pdf - Opening the Black Box of Neural Nets via Information  
https://www.youtube.com/watch?v=ekUWO_pI2M8  
https://www.youtube.com/watch?v=bLqJHjXihK8  

### Oct 9, 2017 - Hacker Dojo 
https://arxiv.org/pdf/1501.00092.pdf - super resolution first paper  
https://arxiv.org/abs/1608.00367 - super resolution second paper  

### Oct 2, 2017 - Hacker Dojo
https://arxiv.org/abs/1604.03901 - Single-Image Depth Perception in the Wild  

### Sept 25, 2017 - Hacker Dojo
https://arxiv.org/pdf/1706.08947.pdf - Exploring generalization in deep networks.  

### Sept 18, 2017 - Hacker Dojo
https://arxiv.org/pdf/1705.02550.pdf - nvidia drone nav  
https://github.com/NVIDIA-Jetson/redtail/wiki - code  

### Sept 11, 2017 - Hacker Dojo
http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.365.5060&rep=rep1&type=pdf - hyperneat ref  
https://arxiv.org/pdf/1609.09106.pdf - Hypernet ref  
http://blog.otoro.net/2016/09/28/hyper-networks/ - blog on hypernet  
https://www.youtube.com/watch?v=-8oyTYViuJ4 - vid on hyperNeat  
http://eplex.cs.ucf.edu/hyperNEATpage/HyperNEAT.html - blog on hyperNeat

### August 28, 2017 - Hacker Dojo
https://arxiv.org/pdf/1708.05344.pdf - SMASH: One-Shot Model Architecture Search through HyperNetworks
https://www.youtube.com/watch?v=79tmPL9AL48 - youtube vid on SMASH  

### August 21, 2017 - Hacker Dojo
https://arxiv.org/pdf/1706.02515.pdf - Self Normalizing Neural Networks - Hochreiter  

### August 14, 2017 - Hacker Dojo  
https://arxiv.org/pdf/1606.01541.pdf - Reinforcement Learning for Dialog Generation - Jurafsky  
https://github.com/liuyuemaicha/Deep-Reinforcement-Learning-for-Dialogue-Generation-in-tensorflow - tensorflow code for same  
https://github.com/jiweil/ - some related code  
https://arxiv.org/pdf/1612.00563.pdf - self critical training for image captioning - RL for text prob.  
  
Some papers referenced by Jurafsky paper 
[1506.05869] A Neural Conversational Model - Vinyals and Le  
https://arxiv.org/abs/1604.04562 - Dialogue generation system - Wen  


### Aug 7, 2017 - Hacker Dojo
https://arxiv.org/pdf/1705.04304.pdf - A Deep Reinforced Model for Abstractive Summarization - socher 

### July 31, 2017 - Hacker Dojo
https://arxiv.org/pdf/1706.01433.pdf - visual interaction networks - deep mind  
https://arxiv.org/pdf/1706.01427.pdf - neural model for relational reasoning - deep mind   


### July 24, 2017  
Guest Speaker - Using FPGA to speed CNN.  
https://arxiv.org/pdf/1703.03130.pdf - A structured self-attentive sentence embedding - Lin and Bengio  
https://github.com/dennybritz/deeplearning-papernotes/blob/master/notes/self_attention_embedding.md (review)  
https://github.com/yufengm/SelfAttentive  code  
https://github.com/Diego999/SelfSent  code  

### July 17, 2017 - Hacker Dojo  
https://arxiv.org/pdf/1706.03762.pdf - attention is all you need - Vaswani  
https://github.com/tensorflow/tensor2tensor/tree/master/tensor2tensor/models  
https://github.com/jadore801120/attention-is-all-you-need-pytorch - easier to read code  
https://arxiv.org/pdf/1607.06450.pdf - layer normalization paper - hinton  
https://www.youtube.com/watch?v=nR74lBO5M3s - google translate paper - youtube video  
https://arxiv.org/pdf/1609.08144.pdf  - google translate paper - 

### July 10, 2017 - Hacker Dojo  
https://arxiv.org/pdf/1706.03762.pdf - attention is all you need - Vaswani  
https://github.com/tensorflow/tensor2tensor/tree/master/tensor2tensor/models  
https://github.com/jadore801120/attention-is-all-you-need-pytorch - easier to read code  
https://arxiv.org/pdf/1607.06450.pdf - layer normalization paper - hinton  


#### Some added references regarding positional encodings
http://www.machinelearning.org/proceedings/icml2006/047_Connectionist_Tempor.pdf - A. Graves, S. Fernandez, F. Gomez, and J. Schmidhuber  
https://www.reddit.com/r/MachineLearning/comments/6jdi87/r_question_about_positional_encodings_used_in/  


### June 26, 2017 - Hacker Dojo
https://arxiv.org/pdf/1705.03122.pdf - convolutional sequence to sequence learning  
https://arxiv.org/pdf/1706.03762.pdf - attention is all you need - Vaswani  
http://www.machinelearning.org/proceedings/icml2006/047_Connectionist_Tempor.pdf - A. Graves, S. Fernandez, F. Gomez, and J. Schmidhuber  


### June 19, 2017 - Hacker Dojo
https://arxiv.org/pdf/1701.02720.pdf - RNN for end to end voice recognition


### June 12, 2017 - Hacker Dojo  
New reinforcement learning results -- Too cool for school.  Watch the video and you'll be hooked.  
https://www.youtube.com/watch?v=2vnLBb18MuQ&feature=em-subs_digest  

http://www.cs.ubc.ca/~van/papers/2017-TOG-deepLoco/index.html - paper  


### May 22, 2017 - Hacker Dojo  
https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/HintonDengYuEtAl-SPM2012.pdf - comparison of RNN and HMM for speech recognition  

### May 15, 2017 - Hacker Dojo  
https://arxiv.org/pdf/1412.6572.pdf -  Explaining and Harnessing Adversarial Examples


### May 1, 2017 - Hacker Dojo  
https://arxiv.org/abs/1704.03453  - The Space of Transferable Adversarial Examples


### Apr 24, 2017 - Hacker Dojo  
https://discourse-production.oss-cn-shanghai.aliyuncs.com/original/3X/1/5/15ba4cef726cab390faa180eb30fd82b693469f9.pdf - Using TPU for data center  


### Apr 17, 2017 - Hacker Dojo
Reservoir Computing by Felix Grezes.
http://www.gc.cuny.edu/CUNY_GC/media/Computer-Science/Student%20Presentations/Felix%20Grezes/Second_Exam_Survey_Felix_Grezes_9_04_2014.pdf  

Slides by Felix Grezes: Reservoir Computing for Neural Networks  
http://www.gc.cuny.edu/CUNY_GC/media/Computer-Science/Student%20Presentations/Felix%20Grezes/Second_Exam_Slides_Felix_Grezes_9-14-2014.pdf
(more at: http://speech.cs.qc.cuny.edu/~felix/ )  

This is a short, very useful backgrounder on randomized projections,  
here used for compressed sensing, in a blog post by Terence Tao  
https://terrytao.wordpress.com/2007/04/13/compressed-sensing-and-single-pixel-cameras/  

and the same story told with illustrations on the Nuit Blanche blog:  
http://nuit-blanche.blogspot.com/2007/07/how-does-rice-one-pixel-camera-work.html  

(BTW http://nuit-blanche.blogspot.com is a tremendous website.)  

If we have time, we may discuss this paper:  
Information Processing Using a Single Dynamical Node as Complex System.  
https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3195233/pdf/ncomms1476.pdf  

### Apr 10, 2017 - Hacker Dojo  

https://arxiv.org/pdf/1603.08678.pdf - Instance-sensitive Fully Convolutional Networks  

https://arxiv.org/pdf/1611.07709.pdf - Fully Convolutional Instance-aware Semantic Segmentation  

### Apr 3, 2017 - Hacker Dojo
https://arxiv.org/pdf/1703.03864.pdf - Sutskever paper on using evolutionary systems for optimizing RL prob  
http://jmlr.csail.mit.edu/papers/volume15/wierstra14a/wierstra14a.pdf - ES paper with algo used in Sutskever paper  


### Mar 27, 2017 - Hacker Dojo
Aurobindo Tripathy will reprise a talk he's going to give at Embedded Summit this year.  His talk will survey recent progress in object detection from RCNN to Single Shot MultiBox Detector and Yolo 9000.


### Mar 20, 2017 - Hacker Dojo  
https://arxiv.org/pdf/1612.05424.pdf - Unsupervised Pixel-level domain adaptation with generative adversarial networks   

## Mar 13, 2017 - Hacker Dojo  
https://arxiv.org/pdf/1701.06547.pdf - adversarial learning for neural dialog generation  

### February 27, 2017 - Hacker Dojo   
https://arxiv.org/pdf/1612.02699.pdf - Deep Supervision with Shape Concepts for Occlusion-Aware 3D Object Parsing  
Zeeshan's slides are in the folder with his name on it.  Along with his descriptions of his own ground-breaking work, he gives an excellent history of efforts to identify 3d objects from 2d images.  


### February 20, 2017 - Hacker Dojo
https://arxiv.org/pdf/1506.07285.pdf  - Ask me anything - Socher  
https://github.com/YerevaNN/Dynamic-memory-networks-in-Theano - Code and implementation notes.  
https://www.youtube.com/watch?v=FCtpHt6JEI8&t=27s - Socher presentation of material  


### February 13, 2017 - Hacker Dojo 
https://arxiv.org/pdf/1701.06538v1.pdf - Outrageously large neural networks  

### February 6, 2017 - Hacker Dojo  

https://arxiv.org/pdf/1505.00387v2.pdf - Highway networks  
https://arxiv.org/pdf/1507.06228.pdf - Also highway networks - different examples   
https://arxiv.org/pdf/1607.03474v3.pdf - Recurrent Highway Networks  


### January 30, 2017 - Hacker Dojo
https://arxiv.org/pdf/1603.03116v2.pdf - Low-rank pass-through RNN's follow-on to unitary rnn
https://github.com/Avmb/lowrank-gru - theano code

### January 23, 2017 - HackerDojo
https://arxiv.org/abs/1612.03242 - Stack Gan Paper  
https://github.com/hanzhanggit/StackGAN - Code  

### January 16, 2017 - Hacker Dojo
https://arxiv.org/pdf/1511.06464v4.pdf - Unitary Evolution RNN
https://github.com/amarshah/complex_RNN - theano code

### January 9, 2017 - Hacker Dojo
Cheuksan Edward Wang Talk  
https://arxiv.org/pdf/1612.04642v1.pdf - rotation invariant cnn  
https://github.com/deworrall92/harmonicConvolutions - tf code for harmonic cnn
http://visual.cs.ucl.ac.uk/pubs/harmonicNets/index.html - blog post by authors

### January 2, 2017 - Hacker Dojo
https://arxiv.org/pdf/1602.02218v2.pdf - using typing to improve RNN behavior  
http://jmlr.org/proceedings/papers/v37/jozefowicz15.pdf - exploration of alternative LSTM architectures  

___________________________________________________________________________________________________________
# ======== 2016 ========

### December 19, 2016 - Hacker Dojo 
https://arxiv.org/pdf/1611.01576.pdf - Socher qRnn paper

### December 12, 2016 - Hacker Dojo 
https://arxiv.org/pdf/1604.02135v2.pdf - latest segmentation fair  
https://github.com/MarvinTeichmann/tensorflow-fcn - code for segmenter   

### December 5, 2016 - Hacker Dojo
https://arxiv.org/pdf/1506.06204.pdf - Object segmentation
https://arxiv.org/pdf/1603.08695v2.pdf - refinement of above segmentation paper  
https://code.facebook.com/posts/561187904071636/segmenting-and-refining-images-with-sharpmask/ - blog post  
https://github.com/facebookresearch/deepmask - torch code for deepmask  


### November 28, 2016 - Hacker Dojo
https://arxiv.org/pdf/1506.01497v3.pdf  
people.eecs.berkeley.edu/~rbg/slides/rbg-defense-slides.pdf - Girshick thesis slides  
Check edge boxes and selective search  
https://arxiv.org/pdf/1406.4729v4.pdf - key part of architecture  
https://github.com/smallcorgi/Faster-RCNN_TF - excellent code  


### November 21, 2016 - Hacker Dojo
https://people.eecs.berkeley.edu/~rbg/papers/r-cnn-cvpr.pdf - RCNN   
https://arxiv.org/pdf/1504.08083v2.pdf - RCNN - first in series  
https://arxiv.org/pdf/1506.01497v3.pdf -  Faster R-CNN   
http://techtalks.tv/talks/rich-feature-hierarchies-for-accurate-object-detection-and-semantic-segmentation/60254/ - video of Girshick talk  


### November 14, 2016 - Hacker Dojo
https://arxiv.org/pdf/1506.02025v3.pdf - Spatial transformer networks   
https://github.com/daviddao/spatial-transformer-tensorflow - tf code for above   

### October 31, 2016 - Hacker Dojo
https://github.com/jazzsaxmafia/show_attend_and_tell.tensorflow - tf code for attention-captioning
http://cs.stanford.edu/people/karpathy/densecap/ - karpathy captioning
https://arxiv.org/pdf/1412.2306v2.pdf - earlier karpathy captioning paper


### October 20, 2016 - Galvanize  
https://webdocs.cs.ualberta.ca/~sutton/book/the-book.html - Deep dive into reinforcement learning - Sutton and Barto - Chapters 1 and 2.  

### Oct 17, 2016 - Hacker Dojo
https://arxiv.org/pdf/1608.06993v1.pdf - DenseNet.  New reigning champion image classifier  
https://github.com/liuzhuang13/DenseNet - lua code  
The DenseNet paper is straight-forward, so we're also going to start on image captioning  

http://www.cs.toronto.edu/~zemel/documents/captionAttn.pdf  
http://kelvinxu.github.io/projects/capgen.html  
http://people.ee.duke.edu/~lcarin/Yunchen9.25.2015.pdf - slides for caption attention

collections of captioning papers. 
https://github.com/kjw0612/awesome-deep-vision#image-captioning - images  
https://github.com/kjw0612/awesome-deep-vision#video-captioning - video  

### Oct 13, 2016 - SF
http://www.mit.edu/~dimitrib/NDP_Encycl.pdf - (early) Bersekas paper on RL, policy and value iteration  
http://www.nervanasys.com/demystifying-deep-reinforcement-learning/?imm_mid=0e2d7e&cmp=em-data-na-na-newsltr_20160420 - blog post on RL. Nice coverage of value iteration  

### Oct 10, 2016 - Hacker Dojo
https://github.com/carpedm20/pixel-rnn-tensorflow - tensorflow code for pixel rnn (and cnn)  

### Sept 19, 2016 - Hacker Dojo  
https://arxiv.org/pdf/1606.05328v2.pdf - Conditional Image Generation with PixelCNN decoders  
https://arxiv.org/pdf/1601.06759v3.pdf - Pixel RNN  
https://drive.google.com/file/d/0B3cxcnOkPx9AeWpLVXhkTDJINDQ/view - wavenet Generative Audio  
https://deepmind.com/blog/wavenet-generative-model-raw-audio/ - wavenet blog  

### Sept 15, 2016 - Galvanize SF
http://www.gitxiv.com/posts/fepYG4STYaej3KSPZ/densely-connected-convolutional-netowork-densenet


### Sept 12, 2016 - Hacker Dojo  
http://arxiv.org/pdf/1410.3916v11.pdf - original memory networks    
https://arxiv.org/pdf/1606.03126v1.pdf - key/value memory augmented nn
http://www.thespermwhale.com/jaseweston/icml2016/icml2016-memnn-tutorial.pdf#page=87 - tutorial on memory networks in language understanding

### August 29, 2016 - Hacker Dojo
https://arxiv.org/pdf/1410.5401v2.pdf - Neural Turing Machines  
https://github.com/carpedm20/NTM-tensorflow  
https://www.youtube.com/watch?v=_H0i0IhEO2g - Alex Graves presentation at microsoft research  
http://www.robots.ox.ac.uk/~tvg/publications/talks/NeuralTuringMachines.pdf - slides for ntm  

### August 25, 2016 - Galvanize (SF)
http://arxiv.org/pdf/1410.3916v11.pdf - original memory networks    
https://arxiv.org/pdf/1606.03126v1.pdf - key/value memory augmented nn
http://www.thespermwhale.com/jaseweston/icml2016/icml2016-memnn-tutorial.pdf#page=87 - tutorial on memory networks in language understanding

### August 22, 2016 - Hacker Dojo
https://arxiv.org/pdf/1605.07648v1.pdf - fractal net - alternative to resnet for ultra-deep convolution
https://github.com/edgelord/FractalNet - tf code  
http://www.gitxiv.com/posts/ibA8QEu8bvBJSDxr9/fractalnet-ultra-deep-neural-networks-without-residuals  

### August 18, 2016 - Galvanize (SF)  
https://arxiv.org/pdf/1602.01783v2.pdf - new RL architecture - deep mind  

Code:
https://github.com/Zeta36/Asynchronous-Methods-for-Deep-Reinforcement-Learning - tf  
https://github.com/miyosuda/async_deep_reinforce - tf  
https://github.com/coreylynch/async-rl - keras (tf)  
https://github.com/muupan/async-rl - chainer (good discussion)  

### August 15, 2016 - Hacker Dojo  
https://arxiv.org/pdf/1607.02533v1.pdf - Hardening deep networks to adversarial examples.  

### August 11, 2016 - Galvanize (SF)
http://www.gitxiv.com/posts/HQJ3F9YzsQZ3eJjpZ/model-free-episodic-control - deep mind gitxiv paper and code on github
https://github.com/sudeepraja/Model-Free-Episodic-Control - other code
https://github.com/ShibiHe/Model-Free-Episodic-Control

### August 8, 2016 - Hacker Dojo  
https://arxiv.org/pdf/1406.2661.pdf - originating paper on generative adversarial net (gan) - goodfellow, bengio  
http://arxiv.org/pdf/1511.06434v2.pdf - deep cnn gan - radford  
https://github.com/Newmu/dcgan_code - theano code for cnn gan - radford  

### August 4, 2016 - Galvanize (SF)
http://www.gitxiv.com/posts/HQJ3F9YzsQZ3eJjpZ/model-free-episodic-control - deep mind gitxiv paper and code on github

### August 1, 2016 - Hacker Dojo
Papers -   
https://drive.google.com/file/d/0B8Dg3PBX90KNWG5KQXNQOFlBLU1JWWVONkN1UFpnbUR6Y0cw/view?pref=2&pli=1 - Using Stochastic RNN for temporal anomaly detection  
https://home.zhaw.ch/~dueo/bbs/files/vae.pdf  - cover math  
https://arxiv.org/pdf/1401.4082v3.pdf - Rezende - Other Original VAE paper  

Code Review -   
https://github.com/oduerr/dl_tutorial/blob/master/tensorflow/vae/vae_demo.ipynb  
https://github.com/oduerr/dl_tutorial/blob/master/tensorflow/vae/vae_demo-2D.ipynb  

### July 28, 2016 - SF
Papers:  
http://arxiv.org/pdf/1410.5401v2.pdf - Neural Turing Machines - Graves et. al.  
https://arxiv.org/pdf/1605.06065v1.pdf - One Shot Learning - DeepMind  

Code:  
http://icml.cc/2016/reviews/839.txt  
https://github.com/brendenlake/omniglot  
https://github.com/tristandeleu/ntm-one-shot  
https://github.com/MLWave/extremely-simple-one-shot-learning  

### July 25, 2016 - Hacker Dojo
Papers - Using VAE for anomaly detection  
https://arxiv.org/pdf/1411.7610.pdf - Stochastic Recurrent Networks  
https://drive.google.com/file/d/0B8Dg3PBX90KNWG5KQXNQOFlBLU1JWWVONkN1UFpnbUR6Y0cw/view?pref=2&pli=1 - Using Stochastic RNN for temporal anomaly detection  
  
### July 21, 2016 - SF
Papers to read:  
http://www.thespermwhale.com/jaseweston/ram/papers/paper_16.pdf  
http://snowedin.net/tmp/Hochreiter2001.pdf - 

Comments / Code  
http://icml.cc/2016/reviews/839.txt  
https://github.com/brendenlake/omniglot  
https://github.com/tristandeleu/ntm-one-shot  
https://github.com/MLWave/extremely-simple-one-shot-learning  
https://www.periscope.tv/hugo_larochelle/1ypJdnPRYEoKW  
  
### July 18, 2016 - Hacker Dojo  
Papers to read:  
http://arxiv.org/pdf/1312.6114v10.pdf - variational autoencoders - U of Amsterdam - Kingma and Welling  
http://arxiv.org/pdf/1310.8499v2.pdf - deep autoregressive networks - deep mind   
 https://arxiv.org/abs/1606.05908 - tutorial on vae
  
Commentaries/Code  
https://jmetzen.github.io/2015-11-27/vae.html - metzen - code and discussion  
http://blog.keras.io/building-autoencoders-in-keras.html - chollet - discusses different autoencoders, gives keras code.  

### June 27, July 11 2016 - Hacker Dojo   
Recurrent network for image generation - Deep Mind   
https://arxiv.org/pdf/1502.04623v2.pdf  
Background and some references cited  
http://blog.evjang.com/2016/06/understanding-and-implementing.html - blog w. code for VAE  
http://arxiv.org/pdf/1312.6114v10.pdf - Variational Auto Encoder  
https://jmetzen.github.io/2015-11-27/vae.html - tf code for variational auto-encoder  
https://www.youtube.com/watch?v=P78QYjWh5sM  

https://arxiv.org/pdf/1401.4082.pdf  - stochastic backpropagation and approx inference - deep mind  
http://www.cs.toronto.edu/~fritz/absps/colt93.html - keep neural simple by minimizing descr length - hinton  
https://github.com/vivanov879/draw - code  


### June 20, 2016 - Peninsula   
Recurrent models of visual attention - Deep Mind   
https://papers.nips.cc/paper/5542-recurrent-models-of-visual-attention.pdf   

### June 23, 29 2016 - SF
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

### June 13, 2016 - TBD, Peninsula
Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning:  
http://arxiv.org/pdf/1602.07261v1.pdf  

### June 9, 2016 - Galvanize
Visualizing and Understanding RNN:  
https://arxiv.org/pdf/1506.02078v2.pdf  

### June 6, 2016 - Hacker Dojo
Google inception paper - origin of 1x1 convolution layers  
http://arxiv.org/pdf/1409.4842v1.pdf  

### June 2, May 26, 2016 - Galvanize
Image segmentation with deep encoder-decoder
https://arxiv.org/pdf/1511.00561.pdf

### May 23, 2016 - Hacker Dojo
Compressed networks, reducing flops by pruning
https://arxiv.org/pdf/1510.00149.pdf
http://arxiv.org/pdf/1602.07360v3.pdf

### May 16, 2016
Word2Vec meets LDA:
http://arxiv.org/pdf/1605.02019v1.pdf - Paper

https://twitter.com/chrisemoody - Chris Moody's twitter with links to slides etc.

http://qpleple.com/topic-coherence-to-evaluate-topic-models/ - writeup on topic coherence


### May 9, 2016
https://arxiv.org/pdf/1603.05027v2.pdf - Update on microsoft resnet - identity mapping  

http://gitxiv.com/posts/MwSDm6A4wPG7TcuPZ/recurrent-batch-normalization - batch normalization w. RNN  


### May 2, 2016
Go playing DQN - AlphaGo
https://gogameguru.com/i/2016/03/deepmind-mastering-go.pdf
https://m.youtube.com/watch?sns=em&v=pgX4JSv4J70 - video of slide presentation on paper
https://en.m.wikipedia.org/wiki/List_of_Go_games#Lee.27s_Broken_Ladder_Game - Handling "ladders" in alphgo
https://en.m.wikipedia.org/wiki/Ladder_(Go) - ladders in go


### April 25, 2016
Deep Residual Learning for Image Recognition    
http://arxiv.org/pdf/1512.03385v1.pdf 

References:
http://arxiv.org/pdf/1603.05027v2.pdf - Identity mapping paper  

Code:  
https://keunwoochoi.wordpress.com/2016/03/09/residual-networks-implementation-on-keras/ - keras code  
https://github.com/ry/tensorflow-resnet/blob/master/resnet.py - tensorflow code  
https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/skflow/resnet.py  


### April 18, 2016 - Batch Normalization   
Playing Atari with Deep Reinforcement Learning  
https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf  
http://gitxiv.com/posts/MwSDm6A4wPG7TcuPZ/recurrent-batch-normalization - Batch Normalization for RNN  


### April 11, 2016
Playing Atari with Deep Reinforcement Learning  
https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf  

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

### April 4, 2016
    
### March 28, 2016
  
### March 21, 2016
  
###  March 14, 2016 
Gated Feedback Recurrent Neural Networks  
https://arxiv.org/pdf/1502.02367v4.pdf)  
  
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

### Reviews  
http://www.computervisionblog.com/2016/06/deep-learning-trends-iclr-2016.html  
https://indico.io/blog/iclr-2016-takeaways/  
