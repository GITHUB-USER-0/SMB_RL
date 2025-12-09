This folder is on the dss2q branch of the project.

This contains attempts and commentary for DQN on Super Mario Bros. (1985).

A brief overview is provided below:

There are 4 different folders containing core code for attempts. The commentary folder contains various musings and was renamed from what I had originally called documentation. 

A more detailed overview is provided below:

**SMB_RL/0_initial_attempt** : an initial attempt. This is from the originally written core template. This served as a proof of concept of being able to run the environment on Rivanna for instance as well as the saving of output frames.
 * scratch.ipynb - interactively working with the core code
 * SMB_iterate.py - originally generated code

**SMB_RL/1_attempt** : a revised attempt that splits the contents into separate class files. Includes the output of a singular long training series.
 * perEpisodeRewards.csv - results from an ~12.8k training, **used for Figure 1 "Rewards per course in Super Mario Bros (1985)"**
 * training_test_old.ipynb - interactive scratch code that was running the training loop
 * helpers.py - a set of helper functions including preprocessing frames, saving images, 
 * DQNAgent.py - DQN agent that takes the actions and incorporates DQN and the replay buffer
 * replay_buffer.py - replay buffer implementation
 * DQN.py - Deep Q Network implementation
 * **savedModels/** : saved .pth models from this run
   * foo5004.pth - a saved model from testing to prove that one could save and load an output pytorch network's weights
   * bar5005.pth - as above
   * 1500_epochs.pth - an example saved output from this iteration
   * 110_epochs.pth - as above
 * **testing/** : testing code to interactively validate the code
   * testDQN.ipynb - test of the corresponding class
   * test_replay_buffer.ipynb - as above
   * prefilling_buffer.ipynb - as above

**SMB_RL/2_chatGPT_revision** : code with substantive revisions from LLM input, though not exclusively LLM input, filenames are as per 1_attempt
 * chatgpt_helpers.py
 * chatgpt_DQNAgent.py
 * chatgpt_replay_buffer.py
 * chatgpt_DQN.py
 * results/  
   * log.csv - a trace of results including additional metadata relative to prior logging attempts, ~5k epochs. A separate copy of this file is contained within Commentary
  
**SMB_RL/3_revised_attempt** : a third attempt that tries to simplify some of the code
 * helpers.py
 * DQNAgent.py
 * replay_buffer.py
 * DQN.py

**SMB_RL/commentary** : a folder containing a variety of notebooks that incorporate thoughts on the process as well as interacting with the provided code.
 * tas_inputs.csv - a processed output file of TAS inputs from "happylee-supermariobros-europe-warps.fm2", this is used in "repitition_in_TAS_inputs.rmd".
 * on TAS and ROMs.ipynb - comments on the potential utility of TAS as sources of curated inputs, validation of the ROM in question, and the unfortunate conclusion that the emulator in use does not align with FCEUX.
 * action_space.ipynb - comments on the action spaces available
 * rewards.ipynb - comments on the reward function in use including its limitations
 * repetition_in_TAS_inputs.Rmd - this is used to make claims about the repetition of inputs across frames which serves as a justification for frame skipping
 * preprocessing frames.ipynb - comments on preprocessing of frames, trimming and the use or absence of grayscaling
 * on different SMB courses.ipynb - comments on different course structrues and the challenges that they could present to an agent. Of note, this contains the *a priori* hypothesis that the puzzle levels might present a challenge for training.
 * **analysis of an attempt** : this corresponds to analysis from 1_attempt
   * perEpisodeRewards.csv - duplicate of SMB_RL/1_attempt/perEpisodeRewards.csv
   * training_round_dss2q_code.png - demonstration of limited learning, with extreme outliers as well as the singular level that reached the flagpole across ~12.8k episodes.
   * per_course_rewards_puzzle_highlight.png - 
   * smb_one_long_training.Rmd
   * PDF_link_to_box.md - a link to a knitted PDF output from `smb_one_long_training.Rmd` as the file would not render properly on GitHub
   * level_categorization.csv - a manual categorization of levels as used in `smb_one_long_training.Rmd`
 * **chatgpt_revision_of_implementation/** : commentary for the 2_chatGPT_revision code
   * log.csv - output from a longer training session (~5k epochs), where some improvement is shown, but it remains unsolved.
   * chatgpt_version_performance_relative_to_level_completion.png - a figure documenting that the rewards of the agent relative to the nominal end of the level, included at the conclusion of the paper.
   * alternative_approach.png - a figure summarizing the performance of the agent over a long training session
   * comments.Rmd - a longer set of comments on the training run of this
   * course_durations.csv - a set of course lengths (widths) used in comments.Rmd to make the chatgpt_version_performance_relative_to_level_completion.png figure

**SMB_RL/embeddedMedia** : various media generated that are referenced in notebooks
 * v0_vs_v3_TAS_inputs.mp4 - a comparison of the same sequence of TAS inputs yielding different outputs in the two different ROM files. This indicates that the differences between ROMs is not purely graphical in nature. See `on TAS and ROMs.ipynb`
 * course1-2_warp_zone.png - demonstration of the agent being at the top of the screen in regular play, which undercuts an assumption of cropping out the top pixels
 * course6-3_postprocessed.png - a frame from a dark background level that presents potential challenges for the use of grayscale preprocessing
 * course6-3_preprocessed_no_grayscale.png - as above
 * course6-3_preprocessed_grayscale.png - as above
 * course6-3_original.png - as above

**SMB_RL/TAS** : TAS inputs
 * happylee-supermariobros-europe-warps.fm2 - TAS inputs from https://tasvideos.org/6622M