// eslint-disable-next-line no-unused-vars
import { BsFillArchiveFill, BsPeopleFill, BsFillBellFill } from 'react-icons/bs';
import React from 'react';
import {
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  LineChart,
  Line,
  PieChart,
  Pie,
  Cell,
} from 'recharts';
import './Dashboard.css';
import { AiOutlineUser, AiOutlineWarning, AiFillAlert, AiOutlineFlag } from 'react-icons/ai';

function Dashboard() {
  /* ──────────────────────────────────────────────────────────
     Synthetic data – replace with real API calls when ready
     ──────────────────────────────────────────────────────────*/
  const summary = {
    total: 1580,
    suspected: 130,
    confirmed: 42,
    reports: 450,
  };

  const trendData = [
    { month: 'Jan', suspected: 10, confirmed: 2 },
    { month: 'Feb', suspected: 20, confirmed: 4 },
    { month: 'Mar', suspected: 25, confirmed: 6 },
    { month: 'Apr', suspected: 18, confirmed: 5 },
    { month: 'May', suspected: 22, confirmed: 8 },
    { month: 'Jun', suspected: 30, confirmed: 10 },
    { month: 'Jul', suspected: 28, confirmed: 7 },
    { month: 'Aug', suspected: 35, confirmed: 9 },
    { month: 'Sep', suspected: 40, confirmed: 12 },
    { month: 'Oct', suspected: 38, confirmed: 11 },
    { month: 'Nov', suspected: 45, confirmed: 14 },
    { month: 'Dec', suspected: 50, confirmed: 16 },
  ];

  const distributionData = [
    { name: 'Legitimate', value: summary.total - (summary.suspected + summary.confirmed) },
    { name: 'Suspected', value: summary.suspected },
    { name: 'Confirmed', value: summary.confirmed },
  ];

  const DIST_COLORS = ['#00b894', '#fdcb6e', '#d63031'];

  return (
    <div className="main-container">
      <div className="main-title">
        <h2 className="dashboard_text">Spam-Account Monitoring</h2>
      </div>

      {/* ── Summary cards ───────────────────────────────── */}
      <div className="main-cards">
        <div className="card">
          <div className="card-inner">
            <h3 className="box_title">Total Users</h3>
            <AiOutlineUser className="card_icon" />
          </div>
          <h1>{summary.total}</h1>
        </div>

        <div className="card">
          <div className="card-inner">
            <h3 className="box_title">Suspected Spammers</h3>
            <AiOutlineWarning className="card_icon" />
          </div>
          <h1>{summary.suspected}</h1>
        </div>

        <div className="card">
          <div className="card-inner">
            <h3 className="box_title">Confirmed Spammers</h3>
            <AiFillAlert className="card_icon" />
          </div>
          <h1>{summary.confirmed}</h1>
        </div>

        <div className="card">
          <div className="card-inner">
            <h3 className="box_title">Spam Reports</h3>
            <AiOutlineFlag className="card_icon" />
          </div>
          <h1>{summary.reports}</h1>
        </div>
      </div>

      {/* ── Charts ─────────────────────────────────────── */}
      <div className="charts">
        {/* Trend line chart */}
        <div className="card graph-card">
          <div className="card-inner">
            <h3 className="box_title">Monthly Spam Trend</h3>
          </div>
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={trendData} margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="month" />
              <YAxis />
              <Tooltip />
              <Legend />
              <Line type="monotone" dataKey="suspected" name="Suspected" stroke="#fdcb6e" />
              <Line type="monotone" dataKey="confirmed" name="Confirmed" stroke="#d63031" />
            </LineChart>
          </ResponsiveContainer>
        </div>

        {/* Distribution pie chart */}
        <div className="card">
          <div className="card-inner">
            <h3 className="box_title">Current Distribution</h3>
          </div>
          <ResponsiveContainer width="100%" height={300}>
            <PieChart>
              <Pie data={distributionData} dataKey="value" outerRadius={120} stroke="#fff">
                {distributionData.map((entry, index) => (
                  <Cell key={`slice-${index}`} fill={DIST_COLORS[index]} />
                ))}
              </Pie>
              <Legend />
              <Tooltip />
            </PieChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  );
}

export default Dashboard;