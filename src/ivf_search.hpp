#pragma once

#include "duckdb.hpp"
#include <vector>
#include <unordered_set> // <-- Added

namespace duckdb {

// --- Updated: Function Data now holds the filter ---
struct IVFSearchFunctionData : public TableFunctionData {
    string index_name;
    int32_t k;
    int32_t nprobe;
    std::vector<float> query_vector;
    
    // Filter Data (Parsed during Bind)
    bool has_filter = false;
    std::unordered_set<int64_t> allowed_ids;
};

// --- State object ---
struct IVFSearchGlobalState : public GlobalTableFunctionState {
    // Index Data
    std::vector<std::vector<float>> centroids;
    
    // Filter Data (Copied from FunctionData)
    bool has_filter = false; 
    std::unordered_set<int64_t> allowed_ids;

    // Results
    struct SearchResult {
        int64_t id;
        float score;
    };
    std::vector<SearchResult> final_results;
    idx_t current_offset = 0;
    bool is_done = false;
};

unique_ptr<FunctionData> BindIVFSearch(ClientContext &context, TableFunctionBindInput &input,
                                       vector<LogicalType> &return_types, vector<string> &names);

unique_ptr<GlobalTableFunctionState> InitIVFSearch(ClientContext &context, TableFunctionInitInput &input);

void ComputeIVFSearch(ClientContext &context, TableFunctionInput &data_p, DataChunk &output);

} // namespace duckdb