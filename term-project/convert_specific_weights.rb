# rails runner convert_to_csv.rb
# http://www.mikeperham.com/2012/05/05/five-common-rails-mistakes/
# Run this for control chart output
#[49, 137, 284].each do |hive_id|
[11].each do |hive_id|
  CSV.open("tmp/hive_#{hive_id}_messages_with_weight.csv", 'wb') do |csv|
    csv << ['time (epoch)', 'weight (kg)']
    Message.where(hive_id: hive_id).order(:occurance_time).find_each do |message| # batches of 1000
      attributes = [message.occurance_time.to_f, message.weight]
      csv << attributes.map { |attr| attr }
    end
  end
end
