#include <iostream>

#include "graph.hpp"
using namespace std;

int main() {
  graph test_graph;
  tensor_ptr in_tensor0, in_tensor1;
  tensor_ptr op0_out_tensor0, op1_out_tensor0, op2_out_tensor0, op3_out_tensor0;
  vector<int> shape0{2, 3}, shape1{3, 4}, shape2{2, 4};
  in_tensor0 = tensor::make(shape0);
  in_tensor1 = tensor::make(shape1);
  op0_out_tensor0 = tensor::make(shape2);
  op1_out_tensor0 = tensor::make(shape2);
  op2_out_tensor0 = tensor::make(shape2);
  op3_out_tensor0 = tensor::make(shape2);
  auto op0 = graph::make("a", {in_tensor0, in_tensor1}, {op0_out_tensor0});
  auto op1 = graph::make("b", {op0->outputs[0]}, {op1_out_tensor0});
  auto op2 = graph::make("c", {op0->outputs[0]}, {op2_out_tensor0});
  auto op3 =
      graph::make("d", {op0->outputs[0], op1->outputs[0]}, {op3_out_tensor0});
  test_graph.add(op0, OpType::Input);
  test_graph.add(op1);
  test_graph.add(op2);
  test_graph.add(op3, OpType::Output);

  return 0;
}