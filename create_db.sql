CREATE DATABASE IF NOT EXISTS plagiarism_app;
   USE plagiarism_app;

   CREATE TABLE IF NOT EXISTS documents (
       id INT AUTO_INCREMENT PRIMARY KEY,
       title VARCHAR(255) NOT NULL,
       content LONGTEXT NOT NULL,
       uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
   );

   CREATE TABLE IF NOT EXISTS embeddings (
       id INT AUTO_INCREMENT PRIMARY KEY,
       document_id INT NOT NULL,
       embedding BLOB NOT NULL,
       FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE CASCADE
   );
