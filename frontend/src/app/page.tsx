/**
 * GroupGen main UI — upload CSV, call POST /generate-groups, render group cards.
 * API contract: see docs/ARCHITECTURE.md and ApiResponse type below.
 */
"use client";

import { useState } from "react";
import Image from "next/image";

type Student = {
  Name: string;
  Gender?: string;
  Motivation?: number;
  Self_Esteem?: number;
  Work_Ethic?: number;
  Learning_Style?: string;
  Diversity?: string;
};

type GroupStats = {
  size: number;
  avg_motivation: number;
  avg_self_esteem?: number;
  avg_work_ethic: number;
  gender_balance?: Record<string, number>;
  learning_styles: string[];
};

type Group = {
  id: number;
  members: Student[];
  stats: GroupStats;
};

type ApiResponse = {
  status: "success" | "success_with_warnings" | string;
  total_students: number;
  total_groups: number;
  target_group_size: number;
  configured_groups: number;
  group_size_summary?: string;
  warnings?: string[];
  groups: Group[];
};

function formatApiError(detail: unknown, fallback: string): string {
  if (detail == null) return fallback;
  if (typeof detail === "string") return detail;
  if (Array.isArray(detail)) {
    return detail
      .map((item) => {
        if (item && typeof item === "object" && "msg" in item) {
          const loc = "loc" in item && Array.isArray(item.loc)
            ? item.loc.join(".")
            : "";
          return loc ? `${loc}: ${item.msg}` : String(item.msg);
        }
        return JSON.stringify(item);
      })
      .join("\n");
  }
  return String(detail);
}

export default function Home() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [groups, setGroups] = useState<Group[] | null>(null);
  const [resultSummary, setResultSummary] = useState<string | null>(null);
  const [warnings, setWarnings] = useState<string[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [groupSize, setGroupSize] = useState(5);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setError(null);
    if (e.target.files && e.target.files[0]) {
      setSelectedFile(e.target.files[0]);
    }
  };

  const handleGenerate = async () => {
    setError(null);
    setWarnings([]);
    setResultSummary(null);
    if (!selectedFile) {
      setError("Please select a CSV file first!");
      return;
    }

    setLoading(true);

    // Browser sends raw CSV; all parsing/validation happens on the Python API.
    const formData = new FormData();
    formData.append("file", selectedFile);

    try {
      const apiUrl = process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:8000";
      const url = `${apiUrl}/generate-groups?group_size=${groupSize}`;

      const response = await fetch(url, {
        method: "POST",
        body: formData,
      });

      if (response.ok) {
        const result: ApiResponse = await response.json();
        if (!Array.isArray(result.groups)) {
          setError("Server returned an unexpected response format.");
          return;
        }
        setGroups(result.groups);
        // Amber banner in UI when fairness swaps could not fix every isolation case.
        setWarnings(result.warnings ?? []);
        const sizeNote = result.group_size_summary
          ? `Group sizes: ${result.group_size_summary}.`
          : "";
        setResultSummary(
          `${result.total_students} students → ${result.total_groups} groups (target ${result.target_group_size} per group). ${sizeNote}`.trim()
        );
      } else {
        let errorData: { detail?: unknown } = {};
        try {
          errorData = await response.json();
        } catch {
          /* non-JSON body */
        }
        setError(
          formatApiError(
            errorData.detail,
            `Request failed (${response.status} ${response.statusText})`
          )
        );
      }
    } catch (err) {
      console.error("API Connection Error:", err);
      setError(
        `Could not connect to backend at ${process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:8000"}`
      );
    } finally {
      setLoading(false);
    }
  };

  const clearResults = () => {
    setGroups(null);
    setResultSummary(null);
    setWarnings([]);
  };

  return (
    <div className="flex h-screen bg-slate-50 font-sans text-slate-800 print:h-auto print:overflow-visible">

      <aside className="w-1/3 min-w-[320px] print:hidden bg-[#e4fdff] border-r border-indigo-100 p-8 flex flex-col shadow-sm z-10">

        <div className="mb-8 flex flex-col items-start">
          <p className="text-medium text-slate-500 font-medium ml-1">AI-Powered Grouping</p>
          <Image
            src="/groupgen-high-resolution-logo.png"
            alt="GroupGen Logo"
            width={180}
            height={10}
            className="mb-2"
            priority
          />
        </div>

        <div className="flex-grow space-y-8">
          <div className="space-y-3">
            <h2 className="font-semibold text-slate-700 uppercase text-xs tracking-wider">1. Upload CSV</h2>
            <div className="border-2 border-dashed border-indigo-100 rounded-xl bg-indigo-50/50 p-6 flex flex-col items-center justify-center text-center hover:border-indigo-300 transition relative cursor-pointer">
              <svg xmlns="http://www.w3.org/2000/svg" className="h-8 w-8 text-indigo-400 mb-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
              </svg>
              <span className="text-sm font-medium text-indigo-600 px-2 truncate max-w-[200px]">
                {selectedFile ? selectedFile.name : "Click to upload"}
              </span>
              <input
                type="file"
                accept=".csv"
                className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
                onChange={handleFileChange}
              />
            </div>
          </div>

          <div className="space-y-3">
            <h2 className="font-semibold text-slate-700 uppercase text-xs tracking-wider">2. Group Size</h2>
            <input
              type="number"
              min="2" max="50"
              value={groupSize}
              onChange={(e) => {
                const n = parseInt(e.target.value, 10);
                if (!Number.isNaN(n)) setGroupSize(n);
              }}
              className="w-full bg-slate-50 border border-slate-200 rounded-lg px-4 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-indigo-500"
            />
          </div>
        </div>

        <div className="pt-6 border-t border-slate-100">
          <button
            onClick={handleGenerate}
            disabled={loading || !selectedFile}
            className={`w-full font-semibold py-3 px-4 rounded-xl shadow-md transition transform active:scale-95
              ${loading || !selectedFile
                ? 'bg-slate-300 text-slate-500 cursor-not-allowed'
                : 'bg-indigo-600 hover:bg-indigo-700 text-white print:hidden'}`}
          >
            {loading ? "Generating..." : "Generate Groups"}
          </button>
        </div>
      </aside>

      <main className="flex-1 bg-slate-50 p-8 overflow-y-auto print:w-full print:p-0 print:bg-white print:overflow-visible print:h-auto">
        {error && (
          <div className="mb-6 bg-red-50 border-l-4 border-red-500 p-4 rounded-md shadow-sm flex justify-between items-start animate-fade-in-down">
            <div className="flex">
              <div className="flex-shrink-0">
                <svg className="h-5 w-5 text-red-500" viewBox="0 0 20 20" fill="currentColor">
                  <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clipRule="evenodd" />
                </svg>
              </div>
              <div className="ml-3">
                <p className="text-sm text-red-700 font-medium whitespace-pre-line">
                  {error}
                </p>
              </div>
            </div>
            <button onClick={() => setError(null)} className="ml-auto pl-3">
              <div className="mx-1.5 -my-1.5">
                <svg className="h-4 w-4 text-red-500 hover:text-red-800" viewBox="0 0 20 20" fill="currentColor">
                  <path fillRule="evenodd" d="M4.293 4.293a1 1 0 011.414 0L10 8.586l4.293-4.293a1 1 0 111.414 1.414L11.414 10l4.293 4.293a1 1 0 01-1.414 1.414L10 11.414l-4.293 4.293a1 1 0 01-1.414-1.414L8.586 10 4.293 5.707a1 1 0 010-1.414z" clipRule="evenodd" />
                </svg>
              </div>
            </button>
          </div>
        )}

        {warnings.length > 0 && (
          <div className="mb-6 bg-amber-50 border-l-4 border-amber-500 p-4 rounded-md shadow-sm">
            <p className="text-sm font-semibold text-amber-800 mb-1">Review recommended</p>
            <ul className="text-sm text-amber-900 list-disc list-inside space-y-1">
              {warnings.map((w) => (
                <li key={w}>{w}</li>
              ))}
            </ul>
          </div>
        )}

        {!groups ? (
          <div className="h-full flex flex-col items-center justify-center text-slate-400 opacity-60">
            <p className="text-lg font-medium">Ready to Group</p>
            <p className="text-sm">Select a file to begin.</p>
          </div>
        ) : (
          <div className="max-w-6xl mx-auto">

            <div className="flex justify-between items-center mb-2">
              <h2 className="text-xl font-bold text-slate-800">Generated Groups</h2>
              <div className="flex gap-3 print:hidden">
                <button
                  onClick={() => window.print()}
                  className="flex items-center gap-2 text-sm font-bold text-indigo-600 border border-indigo-200 bg-indigo-50 px-3 py-1.5 rounded-lg hover:bg-indigo-100 transition"
                >
                  Print Report
                </button>
                <button
                  onClick={clearResults}
                  className="text-sm text-slate-500 hover:text-red-500 px-2"
                >
                  Clear Results
                </button>
              </div>
            </div>
            {resultSummary && (
              <p className="text-sm text-slate-600 mb-6">{resultSummary}</p>
            )}

            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6 print:block print:columns-2">
              {groups.map((group) => {
                const styleCounts = group.members.reduce<Record<string, number>>(
                  (acc, m) => {
                    const s = m.Learning_Style ?? "Unknown";
                    acc[s] = (acc[s] ?? 0) + 1;
                    return acc;
                  },
                  {}
                );
                const styleOrder = ["Visual", "Auditory", "Kinesthetic"];
                const styleBreakdown = styleOrder
                  .filter((s) => styleCounts[s])
                  .map((s) => `${s.charAt(0)} ${styleCounts[s]}`)
                  .join(" · ");
                return (
                <div key={group.id} className="bg-white rounded-xl shadow-sm border border-slate-100 overflow-hidden hover:shadow-md transition print:break-inside-avoid print:mb-6 print:border print:shadow-none">
                  <div className="bg-indigo-50/50 px-4 py-3 border-b border-indigo-50">
                    <div className="flex justify-between items-center print:bg-slate-100 print:border-slate-300">
                      <span className="font-bold text-indigo-900 print:text-black">Group {group.id}</span>
                      <span className="text-xs font-semibold bg-white px-2 py-1 rounded text-slate-500 border border-slate-100 print:border-slate-400">{group.stats.size}</span>
                    </div>
                    <div className="flex flex-wrap gap-2 text-[10px] text-slate-500 uppercase tracking-wider">
                      <div>Motiv: <span className="font-bold text-slate-700">{group.stats.avg_motivation}</span></div>
                      {group.stats.avg_self_esteem != null && (
                        <div>Self: <span className="font-bold text-slate-700">{group.stats.avg_self_esteem}</span></div>
                      )}
                      <div>Work: <span className="font-bold text-slate-700">{group.stats.avg_work_ethic}</span></div>
                    </div>
                    {styleBreakdown && (
                      <div
                        className="mt-1 text-[10px] text-slate-500 uppercase tracking-wider"
                        title="Learning styles in this group (Visual / Auditory / Kinesthetic)"
                      >
                        Styles: <span className="font-bold text-slate-700">{styleBreakdown}</span>
                      </div>
                    )}
                  </div>

                  <ul className="divide-y divide-slate-50">
                    {group.members.map((member) => (
                      <li key={member.Name} className="px-4 py-2 text-sm text-slate-600 flex items-center">
                        <div className="w-6 h-6 rounded-full bg-slate-100 text-slate-500 text-xs flex items-center justify-center mr-3 font-bold select-none">
                          {(member.Name?.charAt(0) ?? "?").toUpperCase()}
                        </div>
                        <span className="truncate">{member.Name ?? "Unknown"}</span>
                      </li>
                    ))}
                  </ul>
                </div>
                );
              })}
            </div>
          </div>
        )}
      </main>
    </div>
  );
}
