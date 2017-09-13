# frozen_string_literal: true

## 9.2
class Two
  DATA = [
    1.22, 1.24, 1.25,
    1.19, 1.17, 1.18,
  ]

  def self.run
    n = DATA.count.to_f

    x = DATA.sum / n
    puts "x = #{x}"

    s2 = (1 / (n - 1).to_f) * DATA.map { |i| (i - x) ** 2 }.sum
    puts "s2 = #{s2}"
  end
end

Two.run
