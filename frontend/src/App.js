import logo from './logo.svg';
import './App.css';
import axios from 'axios';
import React, { useState, useEffect } from 'react';
import VideoUpload from './components/VideoUpload';

function App() {
  return (
    <div className="App">
      <header className="App-header">
        ML Video Translation App
      </header>
      <VideoUpload />
    </div>
  );
}

export default App;
