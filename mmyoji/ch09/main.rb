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

# Two.run

## 9.3
class Three
  DATA = [
    [
      0.3104913,
      0.3304700,
      0.0324358,
      0.8283330,
      0.1727581,
      0.6306326,
      0.7210595,
      0.2451280,
      0.7243750,
      0.8197760,
    ],
    [
      0.2753351,
      0.4359388,
      0.7160295,
      0.7775517,
      0.3251019,
      0.1736013,
      0.0921532,
      0.1318467,
      0.0642188,
      0.8002448,
    ],
    [
      0.3368585,
      0.2513685,
      0.2697405,
      0.1164189,
      0.3085003,
      0.2234060,
      0.9427391,
      0.5800890,
      0.7194922,
      0.8344245,
    ],
    [
      0.4086511,
      0.8016156,
      0.3221239,
      0.8498936,
      0.4362011,
      0.8559286,
      0.9982964,
      0.5540422,
      0.3757575,
      0.1312537,
    ],
    [
      0.4449823,
      0.1457471,
      0.9303545,
      0.1033269,
      0.4415264,
      0.5430776,
      0.8274743,
      0.3946336,
      0.8696082,
      0.6028266,
    ],
  ]

  def self.run
    DATA.each do |set|
      n = set.count.to_f
      average = set.sum / n

      ss = set.map { |i| (i - average) ** 2 }.sum / (n - 1)
      sS = set.map { |i| (i - average) ** 2 }.sum / n

      puts "s^2 = #{ss}"
      puts "S^2 = #{sS}"
    end
  end
end

# Three.run

## 9.8
class Eight
  DATA = [
    171.0,
    167.3,
    170.6,
    178.7,
    162.3
  ]

  def self.run
    n = DATA.count.to_f
    average = DATA.sum / n

    puts "i. #{average}"

    puts "ii."
    exps = []
    DATA.combination(3) do |arr|
      p arr

      avg = arr.sum / 3.0
      exps << avg
      var = arr.map { |i| (i - avg) ** 2 }.sum / 2.0

      puts "sample mean:     #{avg}"
      puts "sample variance: #{var}"
    end

    puts "iii."
    e = exps.map { |i| i / 10.0 }.sum
    puts "E(x) = #{e}"

    # E x bar ^ 2
    v1 = exps.map { |i| (i ** 2 / 10.0) }.sum
    # E(x) ^ 2
    v2 = e ** 2

    puts "v = #{v1 - v2}"
  end
end

Eight.run
