# Generated by Django 5.1.7 on 2025-04-04 14:40

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('market', '0004_category_image_subcategory_image'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='seller',
            name='created_at',
        ),
        migrations.RemoveField(
            model_name='seller',
            name='updated_at',
        ),
        migrations.AddField(
            model_name='seller',
            name='image',
            field=models.ImageField(blank=True, null=True, upload_to='vendors_images/'),
        ),
    ]
