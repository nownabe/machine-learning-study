# frozen_string_literal: true

A = [
  7.97, 7.66, 7.59, 8.44, 8.05,
  8.08, 8.35, 7.77, 7.98, 8.15,
]

B = [
  8.06, 8.27, 8.45, 8.05, 8.51,
  8.14, 8.09, 8.15, 8.16, 8.42,
]

class Calculator
  attr_reader :items
  attr_reader :count

  def initialize items
    @items = items
    @count = items.count.to_f
  end

  # X bar
  def sample_mean
    @sample_mean ||= items.sum / count
  end

  def variance
    @variance ||= items.map { |i| (i - sample_mean) ** 2 }.sum
  end
end

c1 = Calculator.new(A)
c2 = Calculator.new(B)

x1 = c1.sample_mean
x2 = c2.sample_mean

# s^2
s_2 = (1 / (c1.count + c2.count - 2)) * (c1.variance + c2.variance)
s = Math.sqrt(s_2)

puts "x1  = #{x1}"
puts "x2  = #{x2}"
puts "s^2 = #{s_2}"
puts "s   = #{s}"

# t_{0.025}(18) = 2.101
t = 2.101

min = x1 - x2 - t * s * Math.sqrt((1 / c1.count) + (1 / c2.count))
max = x1 - x2 + t * s * Math.sqrt((1 / c1.count) + (1 / c2.count))
puts "range: [#{min}, #{max}]"
