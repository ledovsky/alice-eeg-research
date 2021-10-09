pop_editoptions( 'option_scaleicarms', 0);
EEG = pop_importdata('setname', '2', 'data', 'test2.csv', 'dataformat', 'ascii', 'chanlocs', 'test2.xyz', 'srate', 250);
[ALLEEG EEG CURRENTSET] = pop_newset(ALLEEG, EEG, CURRENTSET, 'gui', 'off');
EEG = pop_editset(EEG, 'srate', [250], 'run', [], 'chanlocs', 'test2.xyz', 'icaweights', 'test2_ica.csv', 'icasphere', 'eye(19)');
EEG = iclabel(EEG);
[ALLEEG EEG] = eeg_store(ALLEEG, EEG, CURRENTSET);
writematrix(EEG.etc.ic_classification.ICLabel.classifications, 'test2_iclabel.csv');