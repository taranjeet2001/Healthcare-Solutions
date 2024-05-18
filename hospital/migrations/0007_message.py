# Generated by Django 4.2.2 on 2024-04-28 16:03

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('hospital', '0006_alter_gallery_category'),
    ]

    operations = [
        migrations.CreateModel(
            name='Message',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('user_message', models.TextField()),
                ('bot_response', models.TextField()),
                ('timestamp', models.DateTimeField(auto_now_add=True)),
            ],
        ),
    ]
