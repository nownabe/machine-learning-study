# frozen_string_literal: true

def make_ngram(n, arr)
  print "#{n}-gram: "
  vector = Hash.new { |h, k| h[k] = 0 }
  arr.each_cons(n).each_with_object(vector) { |c, v| v[c.join] += 1 }.sort_by { |a| a[0] }
end

def print_ngram(ngram)
  puts "[" + ngram.map { |e| e.join(":") }.join(", ") + "]"
end

puts "======== 1 ========"

word = "tattarrattat"

print_ngram(make_ngram(1, word.chars))
print_ngram(make_ngram(2, word.chars))
print_ngram(make_ngram(3, word.chars))

puts
puts "======== 2 ========"

sentence = "A cat sat on the mat."
stop_words = %w[a the on in of]

dic = { "sat" => "sit" }

doc = sentence.split(" ")
doc.map! { |w| w.tr(".", "") }
doc.map(&:downcase)
doc.map! { |w| dic.key?(w) ? dic[w] : w }
doc.delete_if { |w| stop_words.include?(w) }
print_ngram(make_ngram(1, doc))

puts
puts "======== 3 ========"

sentence = "I had a supercalifragilisticexpialidocious time with friends."
puts "[-2=had:1, -1=a:1, +1=time:1, +2=with:1]"
