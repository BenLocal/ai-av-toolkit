use std::sync::{Arc, Mutex};

use rodio::{
    DeviceTrait as _,
    cpal::{
        self, FromSample, InputCallbackInfo, Stream,
        traits::{HostTrait as _, StreamTrait as _},
    },
};
use sherpa_rs::{
    sense_voice::SenseVoiceRecognizer,
    silero_vad::{SileroVad, SileroVadConfig},
};

fn main() {
    list_audio_devices();

    let recognizer = Arc::new(Mutex::new(SileroVadRecognizer::new()));
    let host = cpal::default_host();
    let input_device = match host.default_input_device() {
        Some(device) => device,
        None => {
            println!("No default input device found");
            return;
        }
    };
    let config = input_device.default_input_config().unwrap();

    let sample_format = config.sample_format();
    let stream = match sample_format {
        cpal::SampleFormat::F32 => build_stream::<f32>(&input_device, &config, recognizer.clone()),
        cpal::SampleFormat::I16 => build_stream::<i16>(&input_device, &config, recognizer.clone()),
        cpal::SampleFormat::U16 => build_stream::<u16>(&input_device, &config, recognizer.clone()),
        _ => panic!("Unsupported sample format"),
    };

    stream.play().expect("Failed to play the stream");

    println!("麦克风已激活，按Enter键停止...");
    let mut input = String::new();
    std::io::stdin().read_line(&mut input).unwrap();

    println!("停止麦克风活动");
}
fn build_stream<T>(
    device: &cpal::Device,
    config: &cpal::SupportedStreamConfig,
    recognizer: Arc<Mutex<SileroVadRecognizer>>,
) -> Stream
where
    T: cpal::Sample + cpal::SizedSample,
    f32: FromSample<T>,
{
    let input_sample_rate = config.sample_rate().0;
    let input_channels = config.channels();
    // 16kHz
    let target_sample_rate = 16000;

    let stream = device
        .build_input_stream(
            &config.clone().into(),
            {
                let recognizer = recognizer.clone();
                move |data: &[T], _info: &InputCallbackInfo| {
                    let samples: Vec<f32> = data
                        .iter()
                        .map(|&sample| sample.to_sample::<f32>())
                        .collect();
                    let mono_samples = convert_to_mono(&samples, input_channels);
                    let resampled = if input_sample_rate != target_sample_rate {
                        resample_audio(&mono_samples, input_sample_rate, target_sample_rate)
                    } else {
                        mono_samples
                    };

                    if let Ok(mut recognizer) = recognizer.lock() {
                        process_audio_data(&resampled, &mut *recognizer);
                    }
                }
            },
            move |err| {
                eprintln!("an error occurred on stream: {err}");
            },
            None,
        )
        .unwrap();
    stream
}

fn list_audio_devices() {
    // 获取默认的主机设备管理器
    let host = cpal::default_host();

    // 获取所有输出设备
    println!("Output devices:");
    let output_devices = host.output_devices().expect("Failed to get output devices");
    for (device_index, device) in output_devices.enumerate() {
        println!("  Device #{}: {}", device_index, device.name().unwrap());
    }

    // 获取所有输入设备
    println!("\nInput devices:");
    let input_devices = host.input_devices().expect("Failed to get input devices");
    for (device_index, device) in input_devices.enumerate() {
        println!("  Device #{}: {}", device_index, device.name().unwrap());
    }

    // 获取默认输入设备
    if let Some(default_input_device) = host.default_input_device() {
        println!(
            "\nDefault input device: {}",
            default_input_device.name().unwrap()
        );
    }

    // 获取默认输出设备
    if let Some(default_output_device) = host.default_output_device() {
        println!(
            "Default output device: {}",
            default_output_device.name().unwrap()
        );
    }
}

fn process_audio_data(samples: &[f32], recognizer: &mut SileroVadRecognizer) {
    println!("Processing {} samples at 16kHz", samples.len());
    recognizer.vad.accept_waveform(samples.to_vec());
    while !recognizer.vad.is_empty() {
        let segment = recognizer.vad.front();
        let result = recognizer.recognizer.transcribe(16000, &segment.samples);
        println!("✅ Text: {}", result.text);
        recognizer.vad.pop();
    }
}

fn convert_to_mono(samples: &[f32], channels: u16) -> Vec<f32> {
    if channels == 1 {
        return samples.to_vec();
    }

    let channels = channels as usize;
    let frame_count = samples.len() / channels;
    let mut mono = Vec::with_capacity(frame_count);

    for i in 0..frame_count {
        let mut sum = 0.0;
        for c in 0..channels {
            sum += samples[i * channels + c];
        }
        mono.push(sum / channels as f32);
    }

    mono
}

fn resample_audio(input: &[f32], input_rate: u32, output_rate: u32) -> Vec<f32> {
    if input_rate == output_rate {
        return input.to_vec();
    }

    let ratio = output_rate as f64 / input_rate as f64;
    let output_len = (input.len() as f64 * ratio) as usize;
    let mut output = Vec::with_capacity(output_len);

    for i in 0..output_len {
        let src_idx = i as f64 / ratio;
        let idx = src_idx as usize;

        if idx + 1 < input.len() {
            // 线性插值
            let frac = src_idx - idx as f64;
            let sample = input[idx] * (1.0 - frac as f32) + input[idx + 1] * frac as f32;
            output.push(sample);
        } else if idx < input.len() {
            output.push(input[idx]);
        }
    }

    output
}

pub struct SileroVadRecognizer {
    vad: SileroVad,
    recognizer: SenseVoiceRecognizer,
}

impl SileroVadRecognizer {
    pub fn new() -> Self {
        let config = sherpa_rs::sense_voice::SenseVoiceConfig {
            model: "sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17/model.int8.onnx".into(),
            tokens: "sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17/tokens.txt".into(),
            provider: Some("cpu".into()),
            debug: true,
            ..Default::default()
        };

        let recognizer: SenseVoiceRecognizer = SenseVoiceRecognizer::new(config).unwrap();

        let vad_config = SileroVadConfig {
            model: "silero_vad.onnx".into(),
            debug: true,
            ..Default::default()
        };

        let vad = SileroVad::new(vad_config, 60.0 * 10.0).unwrap();
        SileroVadRecognizer { vad, recognizer }
    }
}
