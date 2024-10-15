import logo from './logo.svg';
import './App.css';
import axios from 'axios';
import React, { useState, useEffect } from 'react';
import VideoUpload from './components/VideoUpload';

function App() {
  const [name, setName] = useState(null);

  const api = axios.create({
    baseURL: process.env.REACT_APP_FLASK_URL, // Flask backend URL
  });
  const getData = () => api.get("/");

  useEffect(() => {
    getData()
      .then((response) => {
        setName(response.data);
      })
      .catch((error) => {
        console.error("Error fetching data:", error);
      });
  }, []);

  return (
    <div className="App">
      <header className="App-header">
        {name && <p>{name.name}</p>}
      </header>
      <VideoUpload />
    </div>
  );
}

export default App;
