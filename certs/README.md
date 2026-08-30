# self-signed CA for dev: openssl req -x509 -newkey rsa:2048 -keyout certs/ca.key -out certs/ca.pem -days 365 -nodes -subj /CN=vsector-ca
# openssl req -newkey rsa:2048 -keyout certs/tls.key -out certs/csr.pem -nodes -subj /CN=vsector
# openssl x509 -req -in certs/csr.pem -CA certs/ca.pem -CAkey certs/ca.key -CAcreateserial -out certs/tls.crt -days 365
