#include <any>
#include <assert.h>
#include <memory>
#include <unordered_map>
#include <utility>
#include <vector>

class tensor;
class op;
using tensor_ptr = std::shared_ptr<tensor>;
using op_ptr = std::shared_ptr<op>;
using any_map = std::unordered_map<std::string, std::any>;

enum class OpType {
  Input,
  Output,
  Internal,
};

class op : std::enable_shared_from_this<op> {
public:
  op(const std::string &name, const std::vector<tensor_ptr> &ins,
     const std::vector<tensor_ptr> &outs, const any_map &attributes)
      : op_name(name), inputs(ins), outputs(outs), attrs(attributes) {}

  std::string op_name;
  std::vector<tensor_ptr> inputs;
  std::vector<tensor_ptr> outputs;
  any_map attrs;
  int64_t op_id = -1;

  op_ptr get_shared_ptr() { return std::shared_ptr<op>(this); }
};

class tensor {
public:
  op *producer_owner{nullptr};
  std::vector<int> dims;
  std::vector<int> strides;
  std::vector<std::pair<int, op_ptr>> uses;

  tensor(op *owner, const std::vector<int> &shape,
         const std::vector<int> &stride = {})
      : producer_owner(owner), dims(shape), strides(stride) {}

  ~tensor() = default;

  void attach_use(op_ptr op, int idx) {
    uses.emplace_back(std::make_pair(idx, op));
  }

  void detach_use(const op_ptr &op) {
    for (auto iter = uses.begin(); iter != uses.end(); ++iter) {
      if (iter->second == op) {
        iter = uses.erase(iter);
      } else {
        ++iter;
      }
    }
  }

  void detach_use(const op_ptr &op, int in_idx) {
    for (auto iter = uses.begin(); iter != uses.end(); ++iter) {
      if (iter->first == in_idx && iter->second == op) {
        iter = uses.erase(iter);
      } else {
        ++iter;
      }
    }
  }

  static tensor_ptr make(const std::vector<int> &shape,
                         const std::vector<int> &stride = {}) {
    return std::make_shared<tensor>(nullptr, shape, stride);
  }
};

class graph {
private:
  std::vector<op_ptr> internal_ops;
  std::vector<op_ptr> input_ops;
  std::vector<op_ptr> output_ops;
  int total_op_num = 0;

public:
  static std::shared_ptr<op> make(const std::string &op_name,
                                  const std::vector<tensor_ptr> &inputs,
                                  const std::vector<tensor_ptr> &outputs,
                                  const any_map &attrs = {}) {
    return std::make_shared<op>(op_name, inputs, outputs, attrs);
  }

  void add(const op_ptr &node, OpType op_type = OpType::Internal) {
    node->op_id = total_op_num;
    for (auto &outs : node->outputs) {
      assert(outs->producer_owner == nullptr ||
             outs->producer_owner == node.get());
      outs->producer_owner = node.get();
    }
    for (unsigned i = 0; i < node->inputs.size(); i++) {
      node->inputs[i]->attach_use(node, i);
    }

    switch (op_type) {
    case OpType::Input: {
      input_ops.emplace_back(node);
      break;
    }
    case OpType::Output: {
      node->op_id = output_ops.size();
      output_ops.emplace_back(node);
      break;
    }
    case OpType::Internal: {
      internal_ops.emplace_back(node);
      break;
    }
    default:
      assert(0 && "no support op type");
    }
    total_op_num++;
  }

  const std::vector<op_ptr> &get_inputs() const { return input_ops; }

  const std::vector<op_ptr> &get_outputs() const { return output_ops; }
};