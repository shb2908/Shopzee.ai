// website/amazon/src/ItemSearchBar.js
import React, { useState, useEffect } from "react";

const ItemSearchBar = () => {
  const [items, setItems] = useState([]); // all items from backend
  const [filteredItems, setFilteredItems] = useState([]); // items after filtering
  const [searchQuery, setSearchQuery] = useState("");
  const [filters, setFilters] = useState({
    category: "",
    condition: "",
    grade: "",
    subject: "",
  });

  // Fetch all items once when the component mounts
  useEffect(() => {
    fetch("http://localhost:8000/products") // Ensure this matches your backend endpoint
      .then((res) => res.json())
      .then((data) => {
        if (data) {
          setItems(data);
          setFilteredItems(data);
        }
      })
      .catch((err) => console.error("Error fetching items:", err));
  }, []);

  // Filter items based on the search query and filters
  const filterItems = (query, filters) => {
    let updatedItems = items;
    if (query) {
      updatedItems = updatedItems.filter((item) =>
        item.name.toLowerCase().includes(query.toLowerCase())
      );
    }
    if (filters.category) {
      updatedItems = updatedItems.filter((item) =>
        item.category.toLowerCase().includes(filters.category.toLowerCase())
      );
    }
    if (filters.condition) {
      updatedItems = updatedItems.filter((item) =>
        item.condition.toLowerCase().includes(filters.condition.toLowerCase())
      );
    }
    if (filters.grade) {
      updatedItems = updatedItems.filter((item) =>
        item.grade.toLowerCase().includes(filters.grade.toLowerCase())
      );
    }
    if (filters.subject) {
      updatedItems = updatedItems.filter((item) =>
        item.subject.toLowerCase().includes(filters.subject.toLowerCase())
      );
    }
    setFilteredItems(updatedItems);
  };

  const handleInputChange = (e) => {
    const query = e.target.value;
    setSearchQuery(query);
    filterItems(query, filters);
  };

  const handleFilterChange = (e) => {
    const { name, value } = e.target;
    const newFilters = { ...filters, [name]: value };
    setFilters(newFilters);
    filterItems(searchQuery, newFilters);
  };

  return (
    <div style={{ padding: "10px", maxWidth: "600px", margin: "auto" }}>
      <input
        type="text"
        placeholder="Search items by name..."
        value={searchQuery}
        onChange={handleInputChange}
        style={{
          width: "100%",
          padding: "10px",
          borderRadius: "4px",
          border: "1px solid #ccc",
          marginBottom: "10px",
        }}
      />
      <div style={{ marginBottom: "10px" }}>
        <input
          type="text"
          name="category"
          placeholder="Category"
          value={filters.category}
          onChange={handleFilterChange}
          style={{
            width: "23%",
            padding: "8px",
            marginRight: "2%",
            border: "1px solid #ccc",
            borderRadius: "4px",
          }}
        />
        <input
          type="text"
          name="condition"
          placeholder="Condition"
          value={filters.condition}
          onChange={handleFilterChange}
          style={{
            width: "23%",
            padding: "8px",
            marginRight: "2%",
            border: "1px solid #ccc",
            borderRadius: "4px",
          }}
        />
        <input
          type="text"
          name="grade"
          placeholder="Grade"
          value={filters.grade}
          onChange={handleFilterChange}
          style={{
            width: "23%",
            padding: "8px",
            marginRight: "2%",
            border: "1px solid #ccc",
            borderRadius: "4px",
          }}
        />
        <input
          type="text"
          name="subject"
          placeholder="Subject"
          value={filters.subject}
          onChange={handleFilterChange}
          style={{
            width: "23%",
            padding: "8px",
            border: "1px solid #ccc",
            borderRadius: "4px",
          }}
        />
      </div>
      <ul style={{ listStyle: "none", padding: 0 }}>
        {filteredItems.map((item) => (
          <li
            key={item.Unique_product_id} // Use Unique_product_id as the key
            style={{
              padding: "10px",
              borderBottom: "1px solid #ddd",
            }}
          >
            <h3>{item.Description}</h3> {/* Adjust based on your data structure */}
            <p>Category: {item.category}</p>
            <p>Condition: {item.condition}</p>
            <p>Grade: {item.grade}</p>
            <p>Subject: {item.subject}</p>
            <p>Price: ${item.Price}</p> {/* Adjust based on your data structure */}
          </li>
        ))}
      </ul>
    </div>
  );
};

export default ItemSearchBar;