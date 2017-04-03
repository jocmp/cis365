# rails runner convert_to_csv.rb
# http://www.mikeperham.com/2012/05/05/five-common-rails-mistakes/
def filter_value(value)
  return 0 if value.nil?
  value
end

CSV.open('tmp/all_attrs.csv', 'wb') do |csv|
  Message.where.not(latitude: 0).where.not(longitude: 0).find_each do |message| # batches of 1000
    attributes = [message.weight, message.humidity, message.temperature,
                  message.occurance_time.to_f, message.latitude, message.longitude]
    csv << attributes.map { |attr| filter_value attr }
  end
end

# CSV.open('tmp/just_weight.csv', 'wb') do |csv|
#   csv << ['weight']
#   Message.find_each do |message|
#     csv << [message.weight, message.occurance_time]
#   end
# end
