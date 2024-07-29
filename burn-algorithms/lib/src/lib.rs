use std::{error::Error, io::Write, os::unix::net::UnixStream};

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

#[cfg(test)]
mod tests {
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
}
