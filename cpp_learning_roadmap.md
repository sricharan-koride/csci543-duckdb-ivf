# C++ Learning Roadmap for DuckDB IVF Extension

This roadmap is tailored specifically to the concepts used in your `csci543-duckdb-ivf` project. It breaks down the C++ knowledge required to understand and maintain this codebase.

## Phase 1: The "Shape" of Data (Containers & Types)
Your code relies heavily on the Standard Template Library (STL). You need to be comfortable with these:

### 1. `std::vector` (The Workhorse)
Almost every file uses this. It's a dynamic array that handles memory for you.
*   **Concepts**: `push_back`, `reserve` (performance critical!), `size()`, `data()` (accessing the raw pointer), `clear()`.
*   **In your code**: Used for `sample_vectors`, `centroids`, `pq_codes`, `flat_data`.
*   **Why it matters**: Vectors are how we store high-dimensional data. `reserve()` is used to prevent expensive reallocations when we know the size in advance.
*   **Exercise**: Write a small program that creates a `vector<float>`, reserves memory for 1000 elements, fills it with random numbers, and then iterates over it to find the average.

### 2. `std::pair` and `std::map`
*   **In your code**: `std::map<int, std::vector<int64_t>> inverted_lists` maps a cluster ID to a list of vector IDs. `std::pair<float, int>` is used to store (distance, index) for sorting.
*   **Concept**: Key-value pairs (`map`) and simple tuples (`pair`).
*   **Exercise**: Create a map that counts the frequency of words in a list of strings. Iterate over the map and print the counts.

### 3. Structs for Data Organization
*   **In your code**: `struct PQCodebook`, `struct Candidate`.
*   **Concept**: Grouping related data together. Unlike classes, structs default to public access, often used for "Plain Old Data" (POD).
*   **Exercise**: Define a `struct Point` with `x`, `y`, `z` coordinates. Write a function that takes a `Point` and returns its distance from the origin.

---

## Phase 2: Memory & Performance (The "Fast" Stuff)
This is a database extension; performance is paramount.

### 1. References (`&`) vs Pointers (`*`)
*   **Crucial**: You see `const std::vector<float>& a` everywhere.
*   **Why**: Passing by value copies the data (slow!). Passing by reference (`&`) just passes a "view" (fast). `const` means "I promise not to change it."
*   **In your code**: `L2SquaredDistance` takes vectors by `const reference`.
*   **Exercise**: Write a function that takes a vector by value and one that takes it by reference. Measure the speed difference when passing a huge vector (e.g., 10 million floats).

### 2. Move Semantics (`std::move`)
*   **In your code**: `sample_vectors.push_back(std::move(row_vector));`
*   **Concept**: Instead of copying data from one variable to another, we "steal" the resources. It's like handing over the keys to a house instead of building a replica house.
*   **Exercise**: Create a vector of vectors. Fill a temporary vector and `push_back` it. Then do the same with `std::move`. Note that the original vector is empty after the move.

### 3. Raw Pointers (`float*`)
*   **In your code**: `const float *cb = pq_codebook.codebooks[m].data();`
*   **Why**: Sometimes we need raw speed or need to interface with C-style APIs (like the K-Means library or low-level memory manipulation).
*   **Concept**: Pointer arithmetic (`ptr + offset`).
*   **Exercise**: Create a `vector<float>`. Get its pointer using `.data()`. Use the pointer to iterate through the array and print values.

---

## Phase 3: Algorithms & Logic
The "brain" of your extension.

### 1. Sorting & Lambdas
*   **In your code**:
    ```cpp
    std::sort(refined.begin(), refined.end(),
              [](const ResultPair &a, const ResultPair &b){ return a.first < b.first; });
    ```
*   **Concept**: `std::sort` sorts a range. The third argument is a **Lambda** (anonymous function) that defines *how* to compare two elements.
*   **Exercise**: Create a vector of `struct Person { string name; int age; }`. Sort them by age using a lambda. Then sort them by name.

### 2. Math & Limits
*   **In your code**: `std::numeric_limits<float>::max()`, `std::min`, `std::abs`.
*   **Concept**: Standard math functions and type properties.
*   **Exercise**: Find the minimum and maximum values in a vector of floats without sorting it.

---

## Phase 4: DuckDB Integration (The "Glue")
How your C++ code talks to the database.

### 1. DuckDB Types
*   **Concepts**: `Value`, `Vector`, `DataChunk`, `LogicalType`.
*   **In your code**: Reading arguments in `CreateIVFIndex`, writing results with `Appender`.
*   **Key Insight**: DuckDB processes data in "vectors" (columns) of chunks, not row-by-row, for speed.

### 2. The `Appender`
*   **In your code**: Used to bulk-insert centroids and inverted lists into tables.
*   **Concept**: A high-performance way to write data, much faster than `INSERT` statements.

### 3. Exception Handling
*   **In your code**: `try { ... } catch (std::exception &e) { ... }`
*   **Concept**: Catching errors (like "table not found") gracefully so the database doesn't crash.
