# frozen_string_literal: true

A = [
  25, 24, 25, 26,
]

B = [
  23, 18, 22, 28,
  17, 25, 19, 16,
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
    @sm ||= items.sum / count
  end

  # 11.50
  def sample_variance
    @sv ||= (1 / (count - 1)) * items.map { |i| (i - sample_mean) ** 2 }.sum
  end
end

c1 = Calculator.new(A)
c2 = Calculator.new(B)

x    = c1.sample_mean
y    = c2.sample_mean
s1_2 = c1.sample_variance
s2_2 = c2.sample_variance

# 11.52
degree_of_freedom =
  begin
    num = ((c1.sample_variance / c1.count) + (c2.sample_variance / c2.count)) ** 2
    den = ((c1.sample_variance / c1.count) ** 2 / (c1.count - 1)) + ((c2.sample_variance / c2.count) ** 2 / (c2.count - 1))
    (num / den).round
  end

puts "x    = #{x}"
puts "y    = #{y}"
puts "s1^2 = #{s1_2}"
puts "s2^2 = #{s2_2}"
puts "v    = #{degree_of_freedom}"

# t_{0.025}(8) = 2.306
t = 2.306

# Confidence Coefficient
# 11.53
min =
  x - y - t * Math.sqrt((s1_2 / c1.count) + (s2_2 / c2.count))
max =
  x - y + t * Math.sqrt((s2_2 / c1.count) + (s1_2 / c2.count))

puts "range: [#{min}, #{max}]"

# x    = 25.0
# y    = 21.0
# s1^2 = 0.6666666666666666
# s2^2 = 17.71428571428571
# v    = 8
# range: [0.4417647498741606, 8.89823102665403]
