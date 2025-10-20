use rodio::{
    DeviceTrait as _,
    cpal::{
        self, Stream,
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
    let stream = device
        .build_input_stream(
            &config.clone().into(),
            move |data: &[T], _: &_| {
                println!("Received audio data: {}", data.len());
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
