use rodio::{
    DeviceTrait as _,
    cpal::{
        self, Sample, Stream,
        traits::{HostTrait as _, StreamTrait as _},
    },
};

fn main() {
    list_audio_devices();

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
        cpal::SampleFormat::F32 => build_stream::<f32>(&input_device, &config),
        cpal::SampleFormat::I16 => build_stream::<i16>(&input_device, &config),
        cpal::SampleFormat::U16 => build_stream::<u16>(&input_device, &config),
        _ => panic!("Unsupported sample format"),
    };

    stream.play().expect("Failed to play the stream");

    println!("麦克风已激活，按Enter键停止...");
    let mut input = String::new();
    std::io::stdin().read_line(&mut input).unwrap();

    println!("停止麦克风活动");
}

fn build_stream<T>(device: &cpal::Device, config: &cpal::SupportedStreamConfig) -> Stream
where
    T: cpal::Sample + cpal::SizedSample,
{
    let input_sample_rate = config.sample_rate().0;
    // 16kHz
    let target_sample_rate = 16000;

    let stream = device
        .build_input_stream(
            &config.clone().into(),
            move |data: &[T], _: &_| {
                let samples: Vec<f32> = data.iter().map(|&sample| sample.to_sample()).collect();
                if input_sample_rate != target_sample_rate {
                    let resampled = resample_audio(&samples, input_sample_rate, target_sample_rate);
                    println!(
                        "Original samples: {}, Resampled to 16kHz: {}",
                        data.len(),
                        resampled.len()
                    );

                    // 这里可以处理重采样后的数据
                    process_audio_data(&resampled);
                } else {
                    println!("Audio already at 16kHz: {} samples", samples.len());
                    process_audio_data(&samples);
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

fn process_audio_data(samples: &[f32]) {
    println!("Processing {} samples at 16kHz", samples.len());
}

fn resample_audio(input: &[f32], input_rate: u32, output_rate: u32) -> Vec<f32> {
    if input_rate == output_rate {
        return input.to_vec();
    }

    let ratio = input_rate as f64 / output_rate as f64;
    let output_len = (input.len() as f64 / ratio) as usize;
    let mut output = Vec::with_capacity(output_len);

    for i in 0..output_len {
        let src_index = i as f64 * ratio;
        let src_index_floor = src_index.floor() as usize;
        let src_index_ceil = (src_index_floor + 1).min(input.len() - 1);

        if src_index_floor >= input.len() {
            break;
        }

        // 线性插值
        let fraction = src_index - src_index_floor as f64;
        let sample = if src_index_floor == src_index_ceil {
            input[src_index_floor]
        } else {
            input[src_index_floor] * (1.0 - fraction as f32)
                + input[src_index_ceil] * fraction as f32
        };

        output.push(sample);
    }

    output
}
