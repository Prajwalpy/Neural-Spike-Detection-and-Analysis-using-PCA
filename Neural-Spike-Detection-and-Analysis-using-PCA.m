%% 1.Load the data

load('08.mat');

%% 2. 

% a. Signal magnitude Calculation
% Peak to peak amplitude is used to detect signal amplitude because it
% represents the maximum voltage swing of action potentials, which is a key characteristic of neural signals.
signal_magnitude = max(wave) - min(wave);
disp(['Signal Magnitude: ', num2str(signal_magnitude)]);

% Result: The signal magnitude is 244, indicating the range of voltage fluctuations in the neural signal.


% b. Signal variance Calculation
% Using RMS as variance metric provides a good measure of signal spread
rms_wave = sqrt(mean(wave.^2));
signal_variance = std(wave);
disp(['Signal Varience: ', num2str(signal_variance)]);

% Result: The signal variance is 11.0321, meaning the signal has a moderate spread around the mean

% c. SNR Calculation
signal_power = mean(wave.^2);
noise_power = var(wave);
SNR = 10 * log10(signal_power / noise_power);
disp(['SNR: ', num2str(SNR)]);

% Result: The SNR is -2.409e-06, suggesting that the signal power is very close to the noise power, making it challenging to distinguish spikes from noise


% d. Threshold for spike detection
threshold = 4 * rms_wave;

% e. Plot raw data with threshold
figure('Position', [100, 100, 600, 300]);
plot(time, wave, 'b', 'LineWidth', 1);
hold on;
yline(threshold, 'r--', 'Threshold', 'LineWidth', 2);
xlabel('Time (s)', 'FontSize', 12);
ylabel('Voltage (µV)', 'FontSize', 12);
title('Raw Neural Data with Threshold', 'FontSize', 14);
legend('Data', 'Threshold', 'FontSize', 10);
grid on;

% f. Extraction of action potentials and noise snippets
[peaks, locs] = findpeaks(wave, 'MinPeakHeight', threshold);

% Define window size for spike extraction
window_size = round(0.0010 * 30000); % 1ms * 30kHz sampling rate

% Extraction of action potential snippets using vectorization
valid_idx = find(locs > window_size & locs <= length(wave) - window_size);
valid_locs = locs(valid_idx);
valid_peaks = peaks(valid_idx);

% Creating index matrix for AP snippets
idx_matrix = bsxfun(@plus, (-window_size:window_size)', valid_locs');
ap_snippets = wave(idx_matrix);

% Isuuse faced: Instead of bsxfun I used, (-window_size:window_size)' + valid_locs;
% But there was error saying arrays have incompatible sizes for this
% operation because valid_locs is is not a row vector the expected dimensions were not matched



% Extracting noise snippets using vectorization
spike_mask = false(size(wave));
for i = 1:length(valid_locs)
    spike_mask(max(1, valid_locs(i)-2*window_size):min(length(wave), valid_locs(i)+2*window_size)) = true;
end
% Using 2* window_size to have a safe margin around dected spikes


% Find valid noise regions
potential_noise = find(~spike_mask);
potential_noise = potential_noise(potential_noise > window_size & potential_noise <= length(wave) - window_size);

% Sample noise locations randomly
num_noise = size(ap_snippets, 2);
noise_locs = potential_noise(randperm(length(potential_noise), num_noise));

% Creating index matrix for noise snippets
noise_matrix = bsxfun(@plus, (-window_size:window_size)', noise_locs');
noise_snippets = wave(noise_matrix);
% Noise is sampled from regions where no action potentials are detected, which ensures that it truly represents the background activity
% The number of noise snippets matches the number of detected spikes, allowing statistical analysis of spike and noise characteristics
% Noise snippets are extracted using the same window size as spikes, making them comparable in terms of shape and duration

% Transpose matrices to match original format
ap_snippets = ap_snippets';
noise_snippets = noise_snippets';

%% 3) PCA-based Spike Sorting

% Prepare data for PCA
all_snippets = [ap_snippets; noise_snippets];
mean_waveform = mean(all_snippets, 1);
centered_snippets = ap_snippets - mean_waveform;
centered_noise = noise_snippets - mean_waveform;

% Compute PCA using SVD
[U, S, V] = svd(centered_snippets, 'econ');


% Project data onto principal components
pc_snippets = centered_snippets * V;
pc_noise = centered_noise * V;

% Perform clustering using k-means
num_clusters = 2;
[idx, centroids] = kmeans(pc_snippets(:,1:3), num_clusters, 'Replicates', 10);

% PCA finds the most important features in the waveform data
% Noise is expected to have a lower variance and remain concentrated in a distinct area
% Action potentials are structured waveforms with high variance, forming separate clusters
% K-means clustering is used to help separate these spikes based on their distribution in the PCA space
% Noise cluster (black) typically occupies a separate, denser region compared to the more dispersed spike clusters

%% 4) Visualization

% 3D PCA plot
figure('Position', [100, 100, 500, 400]);
colors = {'r', 'g'};
subplot(1,2,1);
hold on;
for i = 1:num_clusters
    cluster_points = pc_snippets(idx == i, 1:3);
    scatter3(cluster_points(:,1), cluster_points(:,2), cluster_points(:,3), 20, colors{i}, 'filled', 'DisplayName', ['Cluster ' num2str(i)]);
end
scatter3(pc_noise(:,1), pc_noise(:,2), pc_noise(:,3), 20, 'k','filled', 'DisplayName', 'Noise');
xlabel('PC 1', 'FontSize', 12);
ylabel('PC 2', 'FontSize', 12);
zlabel('PC 3', 'FontSize', 12);
title('Principal Components of Noise and APs', 'FontSize', 12);
legend('Location', 'best', 'FontSize', 10);
grid on;
view(45, 30);

% Plot waveforms
figure('Position', [100, 100, 500, 600]);
time_ms = linspace(-1, 1, size(ap_snippets,2));

for i = 1:num_clusters
    subplot(num_clusters+1, 1, i);
    cluster_waves = ap_snippets(idx == i, :);
    hold on;
    
    % Plot individual waveforms
    plot(time_ms, cluster_waves', 'Color', [0.8 0.8 0.8]);
    
    % Plot mean and standard error
    mean_wave = mean(cluster_waves);
    sem = std(cluster_waves)/sqrt(size(cluster_waves,1));
    
    plot(time_ms, mean_wave, colors{i}, 'LineWidth', 2);
    plot(time_ms, mean_wave + 4*sem, '--', 'Color', colors{i});
    plot(time_ms, mean_wave - 4*sem, '--', 'Color', colors{i});
    
    ylabel('Microvolts');
    title(['Cluster ' num2str(i)]);
    grid on;
end

% Plot noise waveforms
subplot(num_clusters+1, 1, num_clusters+1);
hold on;
plot(time_ms, noise_snippets', 'Color', [0.8 0.8 0.8]);
mean_noise = mean(noise_snippets);
sem_noise = std(noise_snippets)/sqrt(size(noise_snippets,1));
plot(time_ms, mean_noise, 'k', 'LineWidth', 2);
plot(time_ms, mean_noise + 4*sem_noise, 'k--');
plot(time_ms, mean_noise - 4*sem_noise, 'k--');
ylabel('Microvolts');
xlabel('Milliseconds');
title('Noise Waveforms');
grid on;

% Discussion of results:

% The first three subplots represent distinct clusters of action potentials
% Each cluster has a unique waveform shape, suggesting different neuron types
% The black plot represents noise waveforms, which are low amplitude compared to action potential clusters
% The 4x standard error bounds provide a measure of variability in each cluster
% Noise waveforms appear more uniform and lower in amplitude, supporting their distinction from true spikes

%% 5) Analysis of Second Data File, run the individual sections after loading the data
load('08.mat');

% The data files used are 10.mat and 08.mat

% Results comparing both the data
% - The signal magnitude in Data 08 is higher when compared to data 10 which indicates a larger voltage fluctation in the neural activity indicating stronger or more frequent neural spikes
% - The higher variance in Data 08 suggests that there is a greater spread in the signal values which means the dataset has more variation in spike amplitudes. This could be beneficial for spike sorting, as distinct clusters might be more separable
% - The SNR is nearly similar for both datasets, indicating that the relative strength of the signal compared to noise is similar. 
% - Data 08 is cleaner and has more distinguishable neural activity, making it more suitable for accurate spike sorting. Dataset 10 is noisier and requires better filtering.