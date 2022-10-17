#include <functional>
#include <list>
#include <stack>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "graph.hpp"

class op_visit {
public:
  using updater_func = std::function<void(op_visit *, const op_ptr)>;
  using selector_func = std::function<op_ptr(op_visit *)>;
  std::list<op_ptr> to_visit;
  std::unordered_set<int> visited;

public:
  op_visit(selector_func select_next_node_func,
           updater_func update_visit_list_func)
      : select_next_node_(select_next_node_func),
        update_visit_list_(update_visit_list_func) {}

  template <typename function>
  void topology_visit_graph(const graph &graph, function f);

  static updater_func create_updater();

  static op_ptr pop_back_selector(op_visit *v);

  static op_ptr deque_selector(op_visit *v);

  void visit_graph(const graph &g, const std::function<void(op_ptr)> &op_func);

  void visit(const std::function<void(op_ptr)> &op_func);

private:
  // return the next node
  selector_func select_next_node_;
  // after a node has been visited, push sub nodes to 'to_visit' stack
  updater_func update_visit_list_;
};

namespace Vistor {
op_visit post_visitor();

op_visit dfs_visitor();

op_visit bfs_visitor();
} // namespace Vistor
