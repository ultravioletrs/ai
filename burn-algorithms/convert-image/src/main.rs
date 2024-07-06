use image::GenericImageView;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let image_path = &args[1];

    let img_result = image::open(image_path);

    let img = match img_result {
        Ok(img) => img,
        Err(e) => {
            println!("Error loading image: {}", e);
            return;
        }
    };

    let (width, height) = img.dimensions();

    let mut pixel_data: Vec<f32> = Vec::new();

    for y in 0..height {
        for x in 0..width {
            let pixel = img.get_pixel(x, y);
            pixel_data.push(pixel[0] as f32);
        }
    }

    println!(
        "[{}]",
        pixel_data
            .iter()
            .map(|x| format!("{:.1}", x))
            .collect::<Vec<String>>()
            .join(", ")
    );
}
