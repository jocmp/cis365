# rails runner convert_to_csv.rb
# http://www.mikeperham.com/2012/05/05/five-common-rails-mistakes/
CSV.open('tmp/all_attrs.csv', 'wb') do |csv|
  csv << Message.attribute_names
  Message.find_each do |message| # batches of 1000
    csv << message.attributes.values
  end
end

CSV.open('tmp/just_weight.csv', 'wb') do |csv|
  csv << ['weight']
  Message.find_each do |message|
    csv << [message.weight]
  end
end
