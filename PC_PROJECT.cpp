%%writefile main.cpp
#include <iostream>
#include <vector>
#include <chrono>
#include <algorithm>
#include <omp.h>

const int EMPTY = 0;
const int PLAYER_MAX = 1;
const int PLAYER_MIN = 2;

class GameState {
public:
    int board[3][3];
    int currentPlayer;

    GameState() { reset(); }

    void reset() {
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
                board[i][j] = EMPTY;
        currentPlayer = PLAYER_MAX;
    }

    bool isTerminal() const {
        return checkWin(PLAYER_MAX) || checkWin(PLAYER_MIN) || isFull();
    }

    bool isFull() const {
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
                if (board[i][j] == EMPTY) return false;
        return true;
    }

    bool checkWin(int player) const {
        for (int i = 0; i < 3; ++i) {
            if (board[i][0] == player && board[i][1] == player && board[i][2] == player) return true;
            if (board[0][i] == player && board[1][i] == player && board[2][i] == player) return true;
        }
        if (board[0][0] == player && board[1][1] == player && board[2][2] == player) return true;
        if (board[0][2] == player && board[1][1] == player && board[2][0] == player) return true;
        return false;
    }

    int evaluate() const {
        if (checkWin(PLAYER_MAX)) return 10;
        if (checkWin(PLAYER_MIN)) return -10;
        return 0;
    }

    std::vector<GameState> getLegalChildren(int player) const {
        std::vector<GameState> children;
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                if (board[i][j] == EMPTY) {
                    GameState nextState = *this;
                    nextState.board[i][j] = player;
                    nextState.currentPlayer = (player == PLAYER_MAX) ? PLAYER_MIN : PLAYER_MAX;
                    children.push_back(nextState);
                }
            }
        }
        return children;
    }
};

int sequentialAlphaBeta(GameState state, int depth, int alpha, int beta, bool maximizingPlayer, long long &nodeCount) {
    nodeCount++;
    if (depth == 0 || state.isTerminal()) return state.evaluate();

    if (maximizingPlayer) {
        int maxEval = -1000;
        for (const auto& child : state.getLegalChildren(PLAYER_MAX)) {
            int eval = sequentialAlphaBeta(child, depth - 1, alpha, beta, false, nodeCount);
            maxEval = std::max(maxEval, eval);
            alpha = std::max(alpha, eval);
            if (beta <= alpha) break;
        }
        return maxEval;
    } else {
        int minEval = 1000;
        for (const auto& child : state.getLegalChildren(PLAYER_MIN)) {
            int eval = sequentialAlphaBeta(child, depth - 1, alpha, beta, true, nodeCount);
            minEval = std::min(minEval, eval);
            beta = std::min(beta, eval);
            if (beta <= alpha) break;
        }
        return minEval;
    }
}

int parallelAlphaBeta(GameState state, int depth, int alpha, int beta, bool maximizingPlayer, long long &nodeCount) {
    nodeCount++;
    if (depth == 0 || state.isTerminal()) return state.evaluate();

    auto children = state.getLegalChildren(maximizingPlayer ? PLAYER_MAX : PLAYER_MIN);
    if (children.empty()) return state.evaluate();

    if (maximizingPlayer) {
        int maxEval = -1000;
        #pragma omp taskgroup
        {
            for (size_t i = 0; i < children.size(); ++i) {
                if (depth >= 3) {
                    #pragma omp task shared(maxEval, alpha) firstprivate(children, i, depth)
                    {
                        long long local_nodes = 0;
                        int eval = sequentialAlphaBeta(children[i], depth - 1, alpha, beta, false, local_nodes);
                        #pragma omp atomic
                        nodeCount += local_nodes;
                        #pragma omp critical
                        {
                            maxEval = std::max(maxEval, eval);
                            alpha = std::max(alpha, eval);
                        }
                    }
                } else {
                    long long local_nodes = 0;
                    int eval = sequentialAlphaBeta(children[i], depth - 1, alpha, beta, false, local_nodes);
                    nodeCount += local_nodes;
                    maxEval = std::max(maxEval, eval);
                    alpha = std::max(alpha, eval);
                }
            }
        }
        return maxEval;
    } else {
        int minEval = 1000;
        #pragma omp taskgroup
        {
            for (size_t i = 0; i < children.size(); ++i) {
                if (depth >= 3) {
                    #pragma omp task shared(minEval, beta) firstprivate(children, i, depth)
                    {
                        long long local_nodes = 0;
                        int eval = sequentialAlphaBeta(children[i], depth - 1, alpha, beta, true, local_nodes);
                        #pragma omp atomic
                        nodeCount += local_nodes;
                        #pragma omp critical
                        {
                            minEval = std::min(minEval, eval);
                            beta = std::min(beta, eval);
                        }
                    }
                } else {
                    long long local_nodes = 0;
                    int eval = sequentialAlphaBeta(children[i], depth - 1, alpha, beta, true, local_nodes);
                    nodeCount += local_nodes;
                    minEval = std::min(minEval, eval);
                    beta = std::min(beta, eval);
                }
            }
        }
        return minEval;
    }
}

int main() {
    std::cout << "=========================================================\n";
    std::cout << "   Parallel Alpha-Beta Pruning for Game-Tree Search\n";
    std::cout << "   VJTI Mumbai | Department of Information Technology\n";
    std::cout << "=========================================================\n\n";

    GameState initialState;
    int searchDepth = 1000;

    long long seq_nodes = 0;
    auto start_seq = std::chrono::high_resolution_clock::now();
    int seq_score = sequentialAlphaBeta(initialState, searchDepth, -1000, 1000, true, seq_nodes);
    auto end_seq = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> seq_duration = end_seq - start_seq;

    std::cout << "[V1] Sequential Alpha-Beta:\n";
    std::cout << "  - Optimal Score: " << seq_score << "\n";
    std::cout << "  - Nodes Evaluated: " << seq_nodes << "\n";
    std::cout << "  - Execution Time:  " << seq_duration.count() << " ms\n\n";

    long long par_nodes = 0;
    omp_set_num_threads(4);

    auto start_par = std::chrono::high_resolution_clock::now();
    int par_score = 0;
    #pragma omp parallel
    {
        #pragma omp single
        {
            par_score = parallelAlphaBeta(initialState, searchDepth, -1000, 1000, true, par_nodes);
        }
    }
    auto end_par = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> par_duration = end_par - start_par;

    std::cout << "[V3] Adaptive Parallel Alpha-Beta (4 Workers):\n";
    std::cout << "  - Optimal Score: " << par_score << "\n";
    std::cout << "  - Nodes Evaluated: " << par_nodes << "\n";
    std::cout << "  - Execution Time:  " << par_duration.count() << " ms\n\n";

    double speedup = seq_duration.count() / par_duration.count();
    double efficiency = speedup / 4.0 * 100.0;

    std::cout << "=========================================================\n";
    std::cout << "               EXPERIMENTAL PERFORMANCE                  \n";
    std::cout << "=========================================================\n";
    std::cout << "  Speedup S(p):       " << speedup << "x\n";
    std::cout << "  Parallel Efficiency:" << efficiency << "%\n";
    std::cout << "  Minimax Correctness: " << (seq_score == par_score ? "VERIFIED (MATCHED)" : "MISMATCH") << "\n";
    std::cout << "=========================================================\n";

    return 0;
}