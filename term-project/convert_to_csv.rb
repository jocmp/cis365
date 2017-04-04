# rails runner convert_to_csv.rb
# http://www.mikeperham.com/2012/05/05/five-common-rails-mistakes/
def filter_value(value)
  return 0 if value.nil?
  value
end

# Run this for control chart output
CSV.open('tmp/all_attrs.csv', 'wb') do |csv|
  Message.find_each do |message| # batches of 1000
    attributes = [message.weight, message.occurance_time.to_f] #, message.latitude, message.longitude]
    csv << attributes.map { |attr| filter_value attr }
  end
end
