# rails runner convert_to_csv.rb
# http://www.mikeperham.com/2012/05/05/five-common-rails-mistakes/
CSV.open('tmp/all_attrs.csv', 'wb') do |csv|
  csv << Message.attribute_names
  Message.find_each do |message| # batches of 1000
    csv << [message.weight,	message.humidity, message.temperature,
            message.occurance_time, message.latitude, message.longitude]
  end
end

CSV.open('tmp/just_weight.csv', 'wb') do |csv|
  csv << ['weight']
  Message.find_each do |message|
    csv << [message.weight, message.occurance_time]
  end
end
