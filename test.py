from sklearn.impute import SimpleImputer

# Создание объекта SimpleImputer
imputer = SimpleImputer()

# Проверка наличия атрибута keep_empty_features
has_keep_empty_features = hasattr(imputer, 'keep_empty_features')

print(f"Does SimpleImputer have 'keep_empty_features' attribute? {has_keep_empty_features}")

attributes = dir(imputer)

# Выводим атрибуты
print(attributes)