#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <iostream>
#include <map>
#include <optional>
#include <queue>
#include <random>
#include <vector>

// to avoid using int constants as states
enum class State { Ok, Broken, Unknown };

// structure to store edges in the map
struct Edge {
    std::string id;
    int u, v;
    double p;
    State state = State::Unknown;
    uint32_t monteCarloState;
};

// helper structure for candidate heap
struct Candidate {
    double score;
    int idx;
    int from, to;

    bool operator<(const Candidate& other) const { return score < other.score; }
};

using Evaluator = std::function<double(int)>;

std::map<std::string, int> idToIdx;
std::vector<Edge> edges;
std::vector<std::vector<int>> adj;
std::vector<double> reward;

std::vector<bool> visited;
std::vector<int> parent, depth;
std::vector<std::string> parentEdgeId;

std::priority_queue<Candidate> pq;
int current;
double clearCost;

// helpers function to communicate with the jury
void look() {
    std::cout << "LOOK" << std::endl;

    for (size_t i = 0; i < adj[current].size(); ++i) {
        std::string _, id, state;
        std::cin >> _ >> id >> _ >> state;

        if (state == "OK") {
            edges[idToIdx[id]].state = State::Ok;
        } else {
            edges[idToIdx[id]].state = State::Broken;
        }
    }
}

bool move(const std::string& id, int to) {
    std::cout << "MOVE " << id << std::endl;
    std::string res;
    std::cin >> res;
    if (res != "OK") {
        return false;
    }
    current = to;
    return true;
}

bool clear(const std::string& id) {
    std::cout << "CLEAR " << id << std::endl;
    std::string res;
    std::cin >> res;
    if (res != "OK") {
        return false;
    }
    edges[idToIdx[id]].state = State::Ok;
    return true;
}

void end() { std::cout << "END" << std::endl; }

// function that returns the other end of the edge
int other(int idx, int v) { return edges[idx].u + edges[idx].v - v; }

void goTo(int target) {
    int a = current, b = target;
    std::vector<int> down;

    while (depth[a] > depth[b]) {
        move(parentEdgeId[a], parent[a]);
        a = parent[a];
    }

    while (depth[b] > depth[a]) {
        down.push_back(b);
        b = parent[b];
    }

    while (a != b) {
        move(parentEdgeId[a], parent[a]);
        a = parent[a];

        down.push_back(b);
        b = parent[b];
    }

    std::reverse(down.begin(), down.end());

    for (int v : down) {
        move(parentEdgeId[v], v);
    }
}

// find all accessible vertices that haven't been visited yet and put edges into pq
void DFS(int v, const Evaluator& evaluate) {
    visited[v] = true;
    if (parent[v] == -1 || adj[v].size() > 1) {
        look();
    }

    for (auto idx : adj[v]) {
        int u = other(idx, v);
        if (visited[u]) {
            continue;
        }

        if (edges[idx].state == State::Ok) {
            move(edges[idx].id, u);
            parent[u] = v;
            parentEdgeId[u] = edges[idx].id;
            depth[u] = depth[v] + 1;
            DFS(u, evaluate);
            move(edges[idx].id, v);
        } else {
            pq.push(Candidate{evaluate(u), idx, v, u});
        }
    }
}

std::optional<Candidate> findBestCandidate(const Evaluator& evaluate) {
    while (!pq.empty()) {
        Candidate c = pq.top();
        pq.pop();

        if (visited[c.to] || edges[c.idx].state != State::Broken) {
            continue;
        }

        double updated = evaluate(c.to);

        if (std::abs(updated - c.score) > 1e-9) {
            c.score = updated;
            pq.push(c);
            continue;
            // the accessible graph and the evaluation could have changed since
            // we put the candidate into the heap, so we have to recalculate the value.
        }

        if (c.score <= 0) {
            return std::nullopt;
        }

        return c;
    }

    return std::nullopt;
}

void solve(const Evaluator& evaluate) {
    current = 0;
    parent[0] = -1;
    depth[0] = 0;
    DFS(0, evaluate);

    while (true) {
        auto best = findBestCandidate(evaluate);
        if (!best) {
            break;
        }

        goTo(best->from);
        clear(edges[best->idx].id);

        parent[best->to] = best->from;
        parentEdgeId[best->to] = edges[best->idx].id;
        depth[best->to] = depth[best->from] + 1;

        move(edges[best->idx].id, best->to);
        DFS(best->to, evaluate);
    }
    end();
}

double simpleDeterministicEvaluator(int v) { return reward[v] - clearCost; }

double simpleStochasticEvaluator(int v) {
    double res = reward[v] - clearCost;
    for (int u : adj[v]) {
        const Edge& e = edges[u];
        if (!visited[e.v + e.u - v]) {
            res += e.p * reward[e.v + e.u - v];
        }
    }
    return res;
}

double repairAwareStochasticEvaluator(int v) {
    double res = reward[v] - clearCost;
    for (int u : adj[v]) {
        const Edge& e = edges[u];
        if (!visited[e.v + e.u - v]) {
            res += e.p * reward[e.v + e.u - v] +
                   (1.0 - e.p) * std::max(0.0, reward[e.v + e.u - v] - clearCost);
        }
    }
    return res;
}

constexpr double aggressionCoefficient = 1.1;
double aggressiveStochasticEvaluator(int v) {
    double res = reward[v] - clearCost;
    for (int u : adj[v]) {
        const Edge& e = edges[u];
        if (!visited[e.v + e.u - v]) {
            res += aggressionCoefficient * e.p * reward[e.v + e.u - v];
        }
    }
    return res;
}

std::mt19937 rng(67);

std::vector<int> monteCarloVisited;
void monteCarloDFS(int v, double& res, uint32_t bitmask) {
    monteCarloVisited[v] = 1;
    res += reward[v];
    for (int idx : adj[v]) {
        int u = other(idx, v);
        if ((edges[idx].monteCarloState & bitmask) && !visited[u] && !monteCarloVisited[u]) {
            monteCarloDFS(u, res, bitmask);
        }
    }
    monteCarloVisited[v] = 0;
}

constexpr int monteCarloScenarios = 32;
double monteCarloEvaluator(int v) {
    double res = 0.0;
    for (int i = 0; i < monteCarloScenarios; ++i) {
        monteCarloDFS(v, res, 1U << i);
    }
    return res / monteCarloScenarios - clearCost;
}

int main(int argc, char** argv) {
    Evaluator evaluate;
    if (argc == 1 || std::strcmp(argv[1], "s") == 0) {
        evaluate = simpleStochasticEvaluator;
    } else if (std::strcmp(argv[1], "d") == 0) {
        evaluate = simpleDeterministicEvaluator;
    } else if (std::strcmp(argv[1], "as") == 0) {
        evaluate = aggressiveStochasticEvaluator;
    } else if (std::strcmp(argv[1], "ra") == 0) {
        evaluate = repairAwareStochasticEvaluator;
    } else if (std::strcmp(argv[1], "mc") == 0) {
        evaluate = monteCarloEvaluator;
    } else {
        std::cout << "Unknown evaluator requested" << std::endl;
        return 0;
    }
    int n, m;
    std::cin >> n >> m >> clearCost;
    edges.resize(m);
    adj.resize(n);
    visited.resize(n, 0);
    monteCarloVisited.resize(n, 0);
    parent.resize(n, -1);
    parentEdgeId.resize(n);
    reward.resize(n);
    depth.resize(n, -1);
    for (int i = 0; i < n; ++i) {
        std::cin >> reward[i];
    }
    for (int i = 0; i < m; ++i) {
        std::cin >> edges[i].id >> edges[i].u >> edges[i].v >> edges[i].p;
        adj[edges[i].u].push_back(i);
        adj[edges[i].v].push_back(i);
        idToIdx[edges[i].id] = i;
    }
    for (Edge& e : edges) {
        std::bernoulli_distribution dist(e.p);

        for (int i = 0; i < 32; ++i) {
            if (dist(rng)) {
                e.monteCarloState |= (uint32_t{1} << i);
            }
        }
    }
    solve(evaluate);
}