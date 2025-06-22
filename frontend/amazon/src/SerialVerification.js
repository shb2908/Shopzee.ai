import React, { useState } from "react";
import { useParams } from "react-router-dom";
import "./SerialVerification.css";

function SerialVerification() {
  const { Unique_product_id } = useParams();
  const [serialNumber, setSerialNumber] = useState("");
  const [isValid, setIsValid] = useState(null);
  const [message, setMessage] = useState("");
  const [validSerialNumbers, setValidSerialNumbers] = useState([
    "9810066677",
    "654321",
    "123456789",
  ]);
  const [verifiedSerialNumbers, setVerifiedSerialNumbers] = useState([
    "9810088899",
  ]);

  const checkSerialNumber = () => {
    if (verifiedSerialNumbers.includes(serialNumber)) {
      setMessage(
        "This serial number is genuine but has already been verified."
      );
      setIsValid(null);
    } else if (validSerialNumbers.includes(serialNumber)) {
      setIsValid(true);
      setMessage("The serial number is genuine.");
      setValidSerialNumbers(
        validSerialNumbers.filter((sn) => sn !== serialNumber)
      );
      setVerifiedSerialNumbers([...verifiedSerialNumbers, serialNumber]);
    } else {
      setIsValid(false);
      setMessage("The serial number is not genuine.");
    }
  };

  return (
    <div className="serial-verification">
      <h1>Serial Number Verification</h1>
      
      <input
        type="text"
        value={serialNumber}
        onChange={(e) => setSerialNumber(e.target.value)}
        placeholder="Enter Serial Number"
        className="serial-input"
      />
      <button className="button verify" onClick={checkSerialNumber}>
        Verify
      </button>
      {message && (
        <p
          className={`verification-result ${
            isValid === null ? "warning" : isValid ? "valid" : "invalid"
          }`}
        >
          {message}
        </p>
      )}
    </div>
  );
}

export default SerialVerification;
