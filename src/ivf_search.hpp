#pragma once

#include "duckdb.hpp"
#include <vector>
#include <unordered_set> 

namespace duckdb {

struct IVFSearchFunctionData : public TableFunctionData {
    string index_name;
    int32_t k;
    int32_t nprobe;
    std::vector<float> query_vector;
    
    // Filter Data 
    bool has_filter = false;
    std::unordered_set<int64_t> allowed_ids;
    // Free-form WHERE clause pushed from optimizer 
    bool has_where = false;
    string where_clause;
    // Base table name 
    string base_table;
};


struct IVFSearchGlobalState : public GlobalTableFunctionState {
    // Index Data
    std::vector<std::vector<float>> centroids;
    
    // Filter Data
    bool has_filter = false; 
    std::unordered_set<int64_t> allowed_ids;
    // Where-clause pushed from optimizer
    bool has_where = false;
    string where_clause;
    // Base table name to scan for candidates
    string base_table;

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

}