#include "graph_visit.hpp"

template <typename function>
void op_visit::topology_visit_graph(const graph &agraph, function f) {
  for (auto op : agraph.get_inputs()) {
    to_visit.push_back(op);
  }

  while (!to_visit.empty()) {
    op_ptr top = to_visit.front();
    if (visited.find(top->op_id) != visited.end()) {
      to_visit.pop_front();
      continue;
    }
    bool ready = true;
    // Need to iterate backwards because some examples are broken and depend
    // on the partition order
    for (auto it = top->inputs.rbegin(); it != top->inputs.rend(); ++it) {
      if ((*it)->producer_owner != nullptr) {
        op *producer = (*it)->producer_owner;
        if (visited.find(producer->op_id) == visited.end()) {
          // Need to visit first
          to_visit.push_back(producer->get_shared_ptr());
          ready = false;
        }
      }
    }
    if (ready) {
      to_visit.pop_front();
      f(top);
      visited.insert(top->op_id);
    }
  }
}

op_visit::updater_func op_visit::create_updater() {
  std::unordered_map<int, int> pending_count;
  return [pending_count](op_visit *v, const op_ptr &cur_op) mutable {
    for (auto &tensor_ptr : cur_op->outputs) {
      for (size_t i = 0; i < tensor_ptr->uses.size(); ++i) {
        auto use = tensor_ptr->uses[i];
        if (pending_count.find(use.first) == pending_count.end()) {
          pending_count[use.first] = use.second->inputs.size());
        }
        pending_count[use.first]--;
        assert(pending_count[use.first] >= 0);
        if (pending_count[use.first] == 0) {
          v->to_visit.emplace_back(use.second);
        }
      }
    }
  };
}

op_ptr op_visit::pop_back_selector(op_visit *v) {
  auto ret_op = v->to_visit.back();
  v->to_visit.pop_back();
  if (v->visited.find(ret_op->op_id) != v->visited.end()) {
    return ret_op;
  }
  return nullptr;
}

op_ptr op_visit::deque_selector(op_visit *v) {
  auto ret_op = v->to_visit.front();
  v->to_visit.pop_front();
  if (v->visited.find(ret_op->op_id) != v->visited.end()) {
    return ret_op;
  }
  return nullptr;
}

void op_visit::visit(const std::function<void(op_ptr)> &op_func) {
  while (!to_visit.empty()) {
    auto next_op = select_next_node_(this);
    if (!next_op)
      continue;
    op_func(next_op);
    update_visit_list_(this, std::move(next_op));
  }
}

void op_visit::visit_graph(const graph &g,
                           const std::function<void(op_ptr)> &op_func) {
  for (auto in_op : g.get_inputs()) {
    to_visit.emplace_back(in_op);
  }
  visit(op_func);
}

namespace Vistor {
op_visit post_visitor();

op_visit dfs_visitor() {
  return op_visit(op_visit::pop_back_selector, op_visit::create_updater());
}

op_visit bfs_visitor() {
  return op_visit(op_visit::deque_selector, op_visit::create_updater());
}

} // namespace Vistor