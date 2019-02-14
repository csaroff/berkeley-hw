Deep Q Learning
===

Implementation of deep Q learning as described in the berkeley reinforcement learning course
[lectures](https://www.youtube.com/watch?v=nZXC5OdDfs4&index=7&list=PLkFD6_40KJIznC9CDbVTjAF2oyt8_VAe3)

Trained on an i5 & gtx 1080 with 16 GB of RAM.

Training for 100 million frames took ~ 8 days.

Results
---

Unfortunately, I failed to graph the 100 episode reward over time, but I did manage to save the agent at 50 million and
100 million frames.

![50 million frames gif](gifs/BreakoutDeterministic-v4-50M.gif)
*The agent after 50 million frames of training.  Note that the agent was hardcoded to press the "next life"
button.  Since we always ended the episode after one life, it never learned to take that action.*

#### 100 million frames of training.
![100 million frames gif](gifs/BreakoutDeterministic-v4-100M.gif)
*The agent after 100 million frames of training.  The agent didn't actually improve after the first ~75 million frames.
Since we only trained with one life, the agent doesn't seem to know what to do after it dies.*
