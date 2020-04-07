# Getting multi-task GAIL working

Tasks to do:

- Extend sampler to:
   1. do even sampling from all supplied envs;
   2. include an environment ID (either string- or integer-valued) along with
       each observation.
   3. log stats separately for each env (as well as combined stats).
- Discriminator:
   1. Makes sure the model is multi-task (it can take & always receives "flags").
   2. Update the training objective to be multi-task.
- Policy/VF:
   1. Make sure that policy and value function are multi-task (only need to
       pass "flags" in).
   2. Make the RewardModel wrapper multi-task.
   3. Update objective to be multi-task.
