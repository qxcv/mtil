# List of tasks I want to complete

Going to add dates to each of these so that I can remove tasks if they get stale.

Changes to make to GAIL:

- Make GAIL omit noop actions by default (could also trim sequences to omit repeated state/action pairs etc.). (2020-02-29)
- Increase default size of GAIL replay buffer so that it contains maybe ~5 epochs of data instead of just one. (2020-02-29)
- Add a BC loss directly to PPO (and also make sure that noop actions are omitted). (2020-02-29)
- Enable multi-task training for GAIL (once it works really well in single-task setting). (2020-02-29)

Other things to do:

- Do RL on a simple version of the move-to-corner task where it's only necessary for the robot to go to a particular corner, and not interact with any shape. Tune it until it works really well. (2020-02-29)
- Consider simplifying the MILBench tasks so that you can turn down trajectory lengths by 20 to 50%, and decrease resolution (as in total area, not linear dimensions) by 50%+. That will probably require increasing shape and robot sizes, and decreasing default resolution for the LoResStack versions of the environment. This should make both RL and IL easier. (2020-02-29)

Novel approaches to add:

- Do something like LLPfP and use the resulting latent space as a high-level action space for standard IL methods.
- Maybe try the "causal confusion in IL" technique (this will be really annoying).
