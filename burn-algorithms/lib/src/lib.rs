#[cfg(not(target_family = "wasm"))]
use std::os::unix::net::UnixStream;
use std::{error::Error, io::Write};

#[cfg(not(target_family = "wasm"))]
pub fn send_data_via_socket(result: String, path: String) -> Result<(), Box<dyn Error>> {
    let mut stream = match UnixStream::connect(path) {
        Ok(stream) => stream,
        Err(e) => return Err(e.into()),
    };

    let data = result.to_string();
    match stream.write_all(data.as_bytes()) {
        Ok(_) => (),
        Err(e) => return Err(e.into()),
    };

    Ok(())
}

pub fn save_results_to_file(result: String, path: String) -> Result<(), Box<dyn Error>> {
    let path = std::path::Path::new(&path);

    if let Some(parent) = path.parent() {
        if !parent.exists() {
            match std::fs::create_dir_all(parent) {
                Ok(_) => (),
                Err(e) => return Err(e.into()),
            };
        }
    }

    let mut output = match std::fs::File::create(path) {
        Ok(output) => output,
        Err(e) => return Err(e.into()),
    };

    match output.write_all(result.as_bytes()) {
        Ok(_) => (),
        Err(e) => return Err(e.into()),
    };

    Ok(())
}

#[cfg(test)]
mod tests {
    use std::fs;

    use super::*;

    #[test]
    fn send_data_via_socket_doesnt_work() {
        let result = send_data_via_socket("test".to_string(), "test".to_string());
        assert!(result.is_err());
    }

    #[test]
    fn send_data_via_socket_invalid_path() {
        let result = send_data_via_socket("test".to_string(), "/tmp/invalid".to_string());
        assert!(result.is_err());
    }

    #[test]
    fn save_results_to_file_works() {
        let result = save_results_to_file("test".to_string(), "test".to_string());
        assert_eq!(result.is_err(), false);
        fs::remove_file("test").unwrap();
    }

    #[test]
    fn save_results_to_file_invalid_path() {
        let result = save_results_to_file("test".to_string(), "/root/invalid".to_string());
        assert!(result.is_err());
    }
}
