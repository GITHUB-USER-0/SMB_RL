This folder is on the dss2q branch of the project.

This contains attempts and commentary for DQN on Super Mario Bros. (1985).

A brief overview is provided below:

There are 4 different folders containing core code for attempts. The commentary folder contains various musings and was renamed from what I had originally called documentation. A review of training sessions as well as corresponding figure generation is contained within the analysis folder.

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
 * helpers.py - as before
 * DQNAgent.py - as before
 * training_run.ipynb - an example of a training run, minimally configured
 * replay_buffer.py - as before
 * DQN.py - as before
 * SMB_RL\3_revised_attempt\results\2025_12_10__03_13_18 - overnight 
   * log.csv - the result of training data
   * config.json - the configuration of various hyperparameters
 * SMB_RL\3_revised_attempt\results\2025_12_10__03_13_18 - overnight\savedModels 
   * 4000.pth - one of the later models saved


**SMB_RL/commentary** : a folder containing a variety of notebooks that incorporate thoughts on the process as well as interacting with the provided code.
 
 * on TAS and ROMs.ipynb - comments on the potential utility of TAS as sources of curated inputs, validation of the ROM in question, and the unfortunate conclusion that the emulator in use does not align with FCEUX.
 * action_space.ipynb - comments on the action spaces available
 * rewards.ipynb - comments on the reward function in use including its limitations
 * repetition_in_TAS_inputs.Rmd - this is used to make claims about the repetition of inputs across frames which serves as a justification for frame skipping
 * preprocessing frames.ipynb - comments on preprocessing of frames, trimming and the use or absence of grayscaling
 * on different SMB courses.ipynb - comments on different course structrues and the challenges that they could present to an agent. Of note, this contains the a priori hypothesis that the puzzle levels might present a challenge for training.   

**SMB_RL\analysis** : a folder containing consolidated analysis of various training sessions and other elements.
   * combined_analysis.Rmd - combined analysis of training elements with figure generation
   * repetition_in_TAS_inputs.Rmd - an exploration of the repetition of actions in TAS inputs to determine an appropriate amount of action-repetition
   **SMB_RL\analysis\inputs** :
     * log_puzzles_excluded.csv - training log from a not performant training session on all levels with the exception of levels with puzzle elements
     * log_all_levels_no_exclusions.csv - training log from a not performant training session on all levels with an almost identical architecture as the performant 1-1 agent
     * log_performant1-1.csv - training log from a performant DQN network trained on just 1-1
     * 2025-11-24.csv - a training log from that day. The training did not result in a performant model, but did yield key insights
     * tas_inputs.csv - a processed output file of TAS inputs from "happylee-supermariobros-europe-warps.fm2", this is used in "repitition_in_TAS_inputs.rmd".   * sizes.csv - course level sizes (width, height)
	 * level_categorization.csv - a manual categorization of levels as used in analysis
	 * course_durations.csv - a set of course lengths (widths) used in comments.Rmd to make the chatgpt_version_performance_relative_to_level_completion.png figure
  **SMB_RL\analysis\outputs** : a folder containing generated outputs, ie., images and/or pdfs that are used in the manuscript
    * revised_attempt_performance_relative_to_level_completion_excluded_levels.png
    * revised_attempt_performance_relative_to_level_completion_excluded_levels.pdf
    * repetition_in_inputs_TAS.png
    * training_round_dss2q_code.png
    * per_course_rewards_puzzle_highlight.png
    * 1-1_performant.png - demonstration of a performant DQN agent training session, Figure 4.
  **SMB_RL\analysis\getting_world_sizes** : 
    * downloading_pngs.py - code to acquire world sizes as based on external source of level layouts

**SMB_RL/embeddedMedia** : various media generated that are referenced in notebooks
 * v0_vs_v3_TAS_inputs.mp4 - a comparison of the same sequence of TAS inputs yielding different outputs in the two different ROM files. This indicates that the differences between ROMs is not purely graphical in nature. See `on TAS and ROMs.ipynb`
 * course1-2_warp_zone.png - demonstration of the agent being at the top of the screen in regular play, which undercuts an assumption of cropping out the top pixels
 * course6-3_postprocessed.png - a frame from a dark background level that presents potential challenges for the use of grayscale preprocessing
 * course6-3_preprocessed_no_grayscale.png - as above
 * course6-3_preprocessed_grayscale.png - as above
 * course6-3_original.png - as above

**SMB_RL/TAS** : TAS inputs
 * happylee-supermariobros-europe-warps.fm2 - TAS inputs from https://tasvideos.org/6622M