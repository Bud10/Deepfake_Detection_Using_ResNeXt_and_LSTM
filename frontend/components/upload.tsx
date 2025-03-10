"use client";
import { FontAwesomeIcon } from "@fortawesome/react-fontawesome";
import React, { useRef, useState } from "react";
import { faTrash } from "@fortawesome/free-solid-svg-icons";
import Image from "next/image";
import axios from "axios";
import crFile from "../assets/crfile.png";

const FileUploadWithConstraints: React.FC = () => {
  const [file, setFile] = useState<File | null>(null);
  const [error, setError] = useState<string>("");
  const [btnName, setbtnName] = useState<string>("Upload");
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const resultsRef = useRef<HTMLDivElement | null>(null);
  const [showResults, setShowResults] = useState(false);
  const [isInputDisabled, setIsInputDisabled] = useState<boolean>(false);
  const [results, setResults] = useState<any>(null);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFile = e.target.files?.[0];

    // Reset results on new file selection
    setShowResults(false);
    if (selectedFile) {
      const fileSize = selectedFile.size / 1024 / 1024; // File size in MB
      const allowedTypes = ["video/mp4", "video/webm", "video/ogg","video/avi"]; // Add more types if needed

      if (fileSize > 100) {
        setError("File size must be less than 100MB");
        setFile(null);
        return;
      }

      if (!allowedTypes.some((type) => selectedFile.type.startsWith(type))) {
        setError("File must be a video file");
        setFile(null);
        return;
      }

      setError("");
      setFile(selectedFile);
      setIsInputDisabled(true);

      setbtnName("Detect");
    }
  };

  const handleRemoveFile = () => {
    setFile(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = "";
    }
    setbtnName("Upload");
    setError("");
    setShowResults(false);
    setIsInputDisabled(false);
  };

  const handleUploadClick = async () => {
    if (!file) {
      alert("Please select a file first!");
      return;
    }
  
    if (btnName === "Upload") {
      fileInputRef.current?.click();
    } else {
      const formData = new FormData();
      formData.append("file", file);
  
      try {
        const response = await fetch("http://localhost:5000/upload", {
          method: "POST",
          body: formData,
        });
  
        if (!response.ok) {
          throw new Error(`Error: ${response.status} ${response.statusText}`);
        }
  
        const data = await response.json();
        console.log("Backend Response:", data);
        console.log(data.predictions[0].label)
        console.log(data.predictions[0].confidence)
        if (data && data.predictions) {
          setResults({
            result: data.predictions[0].label,
            confidence: data.predictions[0].confidence,
          });
          setShowResults(true);
          setIsInputDisabled(true);
        } else {
          setError("Unexpected response format");
        }
      } catch (error) {
        console.error("Error:", error);
        setError("An error occurred while processing the video. Please try again.");
      }
    }
  };
  

  return (
    <>
      <div className="flex flex-col align-middle justify-center items-center">
        <div className="image">
          {file ? (
            <video controls className="m-2" width={390} height={190}>
              <source src={URL.createObjectURL(file)} type="video/mp4" />
              
              Your browser does not support the video tag.
            </video>
          ) : (
            <Image
              src={crFile}
              width={290}
              height={190}
              alt="KHEC LOGO"
              className="m-2"
            />
          )}
        </div>
        <div className="p-4">
          <label className="block text-gray-700 mb-2">Select a video file only</label>
          <input
            type="file"
            accept="video/*"
            onChange={handleFileChange}
            ref={fileInputRef}
            className="border border-gray-300 p-2 rounded-lg w-220"
            disabled={isInputDisabled}
          />
          {error && <p className="text-red-500 mt-2">{error}</p>}
          {file && (
            <p className="mt-2 text-gray-600">
              Selected File: {file.name} ({(file.size / 1024 / 1024).toFixed(2)} MB)&nbsp;
              <button
                onClick={handleRemoveFile}
                className="bg-red-500 text-white px-4 py-2 mt-2 rounded-lg hover:bg-red-600"
              >
                <FontAwesomeIcon icon={faTrash} />
              </button>
            </p>
          )}
        </div>
        <div>
          <button
            type="button"
            onClick={handleUploadClick}
            className="py-2.5 px-5 me-2 mb-2 w-100 text-sm font-medium text-white focus:outline-none bg-blue-950 rounded-md border border-blue-950 hover:bg-blue-500"
          >
            {btnName}
          </button>
        </div>
        {showResults && results && (
  <div
    ref={resultsRef}
    className="flex flex-col bg-blue-800 p-5 rounded-lg shadow-lg text-white font-medium"
  >
    <span className="text-xl mb-2">
      <strong>Result:</strong>{" "}
      <span className={results.result === "REAL" ? "text-green-400" : "text-red-400"}>
        {results.result}
      </span>
    </span>
    {/* <span className="text-lg mb-2">
              <strong>Accuracy:</strong> <span className="text-yellow-400"></span>
            </span> */}
            <span className="text-lg">
              <strong>Confidence:</strong> <span className="text-red-400">{results.confidence.toFixed(2)}</span>
            </span>
  </div>
)}




      </div>
    </>
  );
};

export default FileUploadWithConstraints;
